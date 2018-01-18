# - * -coding: utf-8

import tensorflow as tf
import numpy as np
import math
from functools import reduce
import jieba
import pandas as pd
import collections
import random
from .logger import Log


class Word2vec():
    def __init__(self, data=None, vocabulary=None, embedding_size=200, num_sampled=10,
                 num_steps=200001, batch_size=128, num_skips=1,
                 skip_window=1, initial_lr=0.000001, valid_size=16,
                 valid_window=100):

        self.vocabulary = vocabulary
        self.vocabulary_size = len(vocabulary)
        self.embedding_size = embedding_size
        self.skip_window = skip_window
        self.num_sampled = num_sampled  # 2-5 for large data set, 5-20 for large data set, take a trade-off -> 10
        self.learning_rate = initial_lr
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.num_skips = num_skips

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
        self.graph = tf.Graph()
        with self.graph.as_default():
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
        config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
        with tf.Session(graph=self.graph, config=config) as sess:

            sess.run(self.init)
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

                feed_dict = {
                    self.inputs: batch_inputs,
                    self.labels: batch_labels
                }
                _, loss_val = sess.run([self.train_op, self.loss],
                                       feed_dict=feed_dict)

                # We perform one update step by evaluating the optimizer op (including it
                # in the list of returned values for session.run()
                _, loss_val = sess.run([self.train_op, self.loss], feed_dict=feed_dict)
                average_loss += loss_val

                if step % 1000 == 0:
                    if step > 0:
                        average_loss /= 1000
                    # The average loss is an estimate of the loss over the last 2000 batches.
                    Log.info('Average loss at step ' + str(step) + ': ' + str(average_loss))
                    Log.info('Global step: ' + str(global_step))
                    average_loss = 0

                # Note that this is expensive (~20% slowdown if computed every 500 steps)
                if step % 10000 == 0:
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
            self.saver.save(sess, 'model/model.ckpt', global_step=self.global_step)


class Segmentation:
    DF_COLUMNS = ['rating', 'comment']

    def __init__(self, raw_data=None, dest_data=None,
                 user_dict='data/dict/user_dict.dic',
                 stop_dict='data/dict/stop_words.dic'):
        self.raw_data = raw_data
        self.dest_data = dest_data
        jieba.load_userdict(user_dict)
        with open(stop_dict, encoding='utf-8') as stops:
            self._stopwords = [word.strip() for word in stops]
            Log.info('stop words no.: {}.'.format(len(self._stopwords)))
        self._df = None

    @property
    def stop_words(self):
        return self._stopwords

    def to_csv(self):
        def segment(sentence):
            def concat(str1, str2):
                return '{} {}'.format(str1.strip(), str2.strip())

            def filter_stop_word(word):
                def is_number(s):
                    try:
                        float(s)
                        return True
                    except ValueError:
                        return False

                stripped = word.strip()

                return len(stripped) != 0 and not stripped in self._stopwords and not is_number(stripped)

            filtered = list(filter(filter_stop_word, jieba.cut(sentence, cut_all=False)))
            if len(filtered) == 0:
                return None
            else:
                return reduce(concat, filtered)

        with open(self.raw_data, encoding='utf-8') as raw:
            data_frames = list()
            for raw_line in raw:
                raw_line = raw_line.lower().strip()
                if len(raw_line) == 0:
                    continue
                try:
                    rating, comment = raw_line[0].strip(), raw_line[1:].strip()
                except Exception as e:
                    Log.warning('{} format invalid'.format(raw_line))
                    continue
                comment = segment(comment)
                if comment is None or len(comment.strip()) == 0:
                    continue
                df = pd.DataFrame([[rating, comment.strip()]], columns=self.DF_COLUMNS)
                data_frames.append(df)

        if len(data_frames) != 0:
            self._df = pd.concat(data_frames, ignore_index=True)
            # save the corpus
            self._df.to_csv(self.dest_data, index=False, encoding='utf-8')


if __name__ == '__main__':
    '''generate corpus'''
    # seg = Segmentation(raw_data='./data/corpus/corpus.dat', dest_data='./data/corpus/corpus.csv')
    # seg.to_csv()

    '''generate vocabulary'''
    # from .vocabulary import Vocabulary
    # vocabulary = Vocabulary('./data/corpus/corpus.csv')
    # np.save('./data/vocabulary.npy', np.array(vocabulary.vocabulary))

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

        return reduce(lambda id1, id2: str(id1) + ' ' + str(id2), map(word2id, words))


    #
    # data = pd.read_csv('./data/corpus/corpus_sample_0.csv')
    # data.comment = data.comment.apply(lambda x: str(x).split()).apply(word2id)
    # data.to_csv('corpus_in_id1.csv', index=False, encoding='utf-8')
    # sentences_in_ids = data.tolist()

    data = pd.read_csv('corpus_in_id.csv')
    data = data.comment.apply(str.split).apply(lambda l: list(map(lambda e: int(e), l)))

    # time expensive
    data = reduce(lambda l1, l2: l1.extend(l2) or l1, data.tolist())
    Log.info('data len: {}'.format(len(data)))

    word2vec = Word2vec(data=data, vocabulary=vocabulary)
    word2vec.train()
