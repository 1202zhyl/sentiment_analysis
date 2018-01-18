# - * -coding: utf-8

'''
pc-00$ python distributed.py --job_name=ps --task_index=0
pc-00$ python distributed.py --job_name=worker --task_index=0
pc-01$ python distributed.py --job_name=worker --task_index=1
pc-02$ python distributed.py --job_name=worker --task_index=2
pc-03$ python distributed.py --job_name=worker --task_index=3
'''

import tensorflow as tf
import numpy as np
import math
from functools import reduce
import pandas as pd
import collections
import random
from .logger import Log


class Word2vec():
    def __init__(self, server, cluster, data=None, vocabulary=None, embedding_size=200, num_sampled=10,
                 num_steps=500001, batch_size=128, num_skips=1,
                 skip_window=1, initial_lr=0.000001, valid_size=16,
                 valid_window=100):

        self.server = server
        self.vocabulary = vocabulary
        self.vocabulary_size = len(vocabulary)
        self.embedding_size = embedding_size
        self.skip_window = skip_window
        self.num_sampled = num_sampled  # 2-5 for large data set, 5-20 for large data set, take a trade-off -> 10
        self.learning_rate = initial_lr
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.num_skips = num_skips
        self.cluster = cluster
        # Random set of words to evaluate similarity on.
        # Only pick dev samples in the head of the distribution.
        self.valid_size = valid_size
        self.valid_examples = np.random.choice(valid_window, valid_size, replace=False)
        self.trained_words = 0
        self.data_index = 0
        self.words_to_train = self.batch_size * self.num_steps
        self.data = data  # [id1, id2, id3, ...]
        Log.info('all words count: {}'.format(len(self.data)))
        self.build_graph()

    def build_graph(self):
        with tf.device(tf.train.replica_device_setter(
                worker_device='/job:worker/task:%d' % FLAGS.task_index,
                cluster=self.cluster)):
            self.inputs = tf.placeholder(tf.int32, shape=[self.batch_size])
            self.labels = tf.placeholder(tf.int32, shape=[self.batch_size, 1])
            self.valid_dataset = tf.constant(self.valid_examples, dtype=tf.int32)

            self.embedding_dict = tf.Variable(
                tf.random_uniform([self.vocabulary_size, self.embedding_size], -1.0, 1.0), name='embedding_dict'
            )
            self.nce_weight = tf.Variable(tf.truncated_normal([self.vocabulary_size, self.embedding_size],
                                                              stddev=1.0 / math.sqrt(self.embedding_size)),
                                          name='nce_weight')
            self.nce_biases = tf.Variable(tf.zeros([self.vocabulary_size]), name='nce_biases')

            # embedding shape: (batch_size, embedding_size)
            self.embedding = tf.nn.embedding_lookup(self.embedding_dict, self.inputs)

            self.loss = tf.reduce_mean(
                tf.nn.nce_loss(
                    weights=self.nce_weight,
                    biases=self.nce_biases,
                    labels=self.labels,
                    inputs=self.embedding,
                    num_sampled=self.num_sampled,
                    num_classes=self.vocabulary_size
                ), name='nce_loss'
            )

            self.global_step = tf.get_variable('global_step',
                                               [],
                                               initializer=tf.constant_initializer(0),
                                               trainable=False)

            self.learning_rate = self.learning_rate * tf.maximum(
                0.0001, 1.0 - tf.cast(self.trained_words, tf.float32) / self.words_to_train)

            self.train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss,
                                                                                              global_step=self.global_step)
            norm = tf.sqrt(tf.reduce_sum(tf.square(self.embedding_dict), 1, keep_dims=True))

            self.normalized_embeddings = tf.divide(self.embedding_dict, norm, name='normed_embedding')

            valid_embeddings = tf.nn.embedding_lookup(
                self.normalized_embeddings, self.valid_dataset)
            self.similarity = tf.matmul(valid_embeddings, self.normalized_embeddings, transpose_b=True)
            # Add variable initializer.
            self.init = tf.global_variables_initializer()
            self.saver = tf.train.Saver()

    def generate_batch(self):
        assert self.batch_size % self.num_skips == 0
        assert self.num_skips <= 2 * self.skip_window

        batch = np.ndarray(shape=([self.batch_size]), dtype=np.int32)
        labels = np.ndarray(shape=(self.batch_size, 1), dtype=np.int32)
        span = 2 * self.skip_window + 1  # [ skip_window target skip_window ]
        buffer = collections.deque(maxlen=span)
        if self.data_index + span > len(self.data):
            self.data_index = 0
        buffer.extend(self.data[self.data_index:self.data_index + span])
        self.data_index += span
        for i in range(self.batch_size // self.num_skips):
            context_words = [w for w in range(span) if w != self.skip_window]
            words_to_use = random.sample(context_words, self.num_skips)
            for j, context_word in enumerate(words_to_use):
                batch[i * self.num_skips + j] = buffer[self.skip_window]
                labels[i * self.num_skips + j, 0] = buffer[context_word]
            if self.data_index == len(self.data):
                for word in data[:span]:
                    buffer.append(word)
                self.data_index = span
            else:
                buffer.append(self.data[self.data_index])
                self.data_index += 1
        # Backtrack a little bit to avoid skipping words in the end of a batch
        self.data_index = (self.data_index + len(self.data) - span) % len(self.data)

        return batch, labels

    def train(self):

        sv = tf.train.Supervisor(is_chief=(FLAGS.task_index == 0),
                                 global_step=self.global_step,
                                 logdir='/home/deployer/youlei/projects/sentiment_analysis/checkpoint',
                                 saver=self.saver,
                                 save_model_secs=7200,
                                 init_op=self.init)

        sess_config = tf.ConfigProto(log_device_placement=True,
                                     device_count={'CPU': 8},
                                     device_filters=['/job:ps', '/job:worker/task:%d' % FLAGS.task_index])
        with sv.prepare_or_wait_for_session(server.target, config=sess_config) as sess:
            Log.info('Initialized')
            average_loss = 0

            for step in range(0, self.num_steps):
                batch_inputs, batch_labels = self.generate_batch()
                if batch_inputs is None or batch_labels is None or len(batch_inputs) == 0 or len(batch_labels) == 0:
                    continue
                feed_dict = {
                    self.inputs: batch_inputs,
                    self.labels: batch_labels
                }
                _, loss_val, global_step = sess.run([self.train_op, self.loss, self.global_step],
                                                    feed_dict=feed_dict)

                # We perform one update step by evaluating the optimizer op (including it
                # in the list of returned values for session.run()
                _, loss_val = sess.run([self.train_op, self.loss], feed_dict=feed_dict)
                average_loss += loss_val
                self.trained_words += len(batch_inputs)

                if step % 1000 == 0:
                    if step > 0:
                        average_loss /= 1000
                    # The average loss is an estimate of the loss over the last 2000 batches.
                    Log.info('Average loss at step ' + str(step) + ': ' + str(average_loss))
                    Log.info('Global_step: ' + str(global_step))
                    average_loss = 0

                # Note that this is expensive (~20% slowdown if computed every 500 steps)
                if step % 100000 == 0:
                    sim = self.similarity.eval()
                    for i in range(self.valid_size):
                        valid_word = self.vocabulary[self.valid_examples[i]]
                        top_k = 8  # number of nearest neighbors
                        nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                        log_str = 'Nearest to %s:' % valid_word
                        for k in range(top_k):
                            close_word = self.vocabulary[nearest[k]]
                            log_str = '%s %s,' % (log_str, close_word)
                        Log.info(log_str)
                    Log.info('-----------------------------------------------------')
            self.final_embeddings = self.normalized_embeddings.eval()
            np.save('data/vocabulary_matrix.npy', self.final_embeddings)
            sv.stop()


if __name__ == '__main__':
    # cluster specification
    parameter_servers = ['10.10.0.50:2223', '10.10.0.51:2223']
    workers = ['10.10.0.50:2222',
               '10.10.0.51:2222',
               '10.10.0.52:2222',
               '10.10.0.53:2222',
               '10.10.0.179:2222',
               '10.10.0.182:2222']
    cluster = tf.train.ClusterSpec({'ps': parameter_servers, 'worker': workers})

    # input flags
    tf.app.flags.DEFINE_string('job_name', '', 'Either "ps" or "worker"')
    tf.app.flags.DEFINE_integer('task_index', 0, 'Index of task within the job')
    FLAGS = tf.app.flags.FLAGS

    # start a server for a specific task
    server = tf.train.Server(
        cluster,
        job_name=FLAGS.job_name,
        task_index=FLAGS.task_index)

    if FLAGS.job_name == 'ps':
        Log.info('ps')
        server.join()
    elif FLAGS.job_name == 'worker':
        Log.info('worker {}'.format(FLAGS.task_index))
        data = pd.read_csv('./data/corpus/corpus_{}.csv'.format(FLAGS.task_index))
        vocabulary = np.load('./data/vocabulary.npy').tolist()
        vocabulary = [(w.encode('UTF-8')).decode('UTF-8') for w in vocabulary]

        vocabulary_size = len(vocabulary)


        def word2id(words):
            def word2id(word):
                try:
                    input_id = vocabulary.index(word)
                except Exception as e:
                    input_id = vocabulary_size - 1
                return input_id

            return list(map(word2id, words))


        df_in_id = data.comment.apply(lambda x: str(x).split()).apply(word2id)

        df_in_id.to_csv('corpus_in_id.csv', index=False, encoding='utf-8')
        sentences_in_ids = df_in_id.tolist()

        # time expensive
        data = reduce(lambda l1, l2: l1.extend(l2) or l1, sentences_in_ids)

        word2vec = Word2vec(cluster=cluster, server=server, data=data, vocabulary=vocabulary)
        word2vec.train()
