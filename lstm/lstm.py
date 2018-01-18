# -*- coding: utf-8

import numpy as np
import tensorflow as tf
import datetime
import pandas as pd

from random import randint

numDimensions = 200
n_steps = 150
batch_size = 64
lstmUnits = 128
iterations = 200000
numClasses = 3


def getBatch(df):
    labels = np.zeros([batch_size, numClasses])
    arr = np.zeros([batch_size, n_steps])
    for i in range(batch_size):
        try:
            rating = int(df.iloc[i].rating)
        except Exception as e:
            rating = 3

        if 0 <= rating <= 2:
            rating = 0
        elif 2 < rating <= 3:
            rating = 1
        else:
            rating = 2
        labels[i, rating] = 1
        input = np.array(list(map(int, df.iloc[i].comment.split()))[0:n_steps])
        if len(input) < n_steps:
            input = np.pad(input, (0, n_steps - len(input)), 'constant', constant_values=69999)
        arr[i] = input
    return arr, labels


vocabulary = np.load('vocabulary.npy')
print('Loaded vocabulary!')
vocabulary = vocabulary.tolist()  # Originally loaded as numpy array
vocabulary = [(w.encode('UTF-8')).decode('UTF-8') for w in vocabulary]  # Encode words as UTF-8
vocabulary_matrix = np.load('vocabularyMatrix.npy')
print('Loaded vocabularyMatrix!')

tf.reset_default_graph()

labels = tf.placeholder(tf.float32, [batch_size, numClasses])
input_data = tf.placeholder(tf.int32, [batch_size, n_steps])
keep_probs = tf.placeholder(tf.float32, [4])
is_training = tf.placeholder(tf.bool)

# inputs = tf.Variable(tf.zeros([batch_size, n_steps, numDimensions]), dtype=tf.float32)
inputs = tf.nn.embedding_lookup(vocabulary_matrix, input_data)

cells = []

for i in range(4):
    cell = tf.contrib.rnn.BasicLSTMCell(lstmUnits)
    if is_training is not None:
        cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=keep_probs[3 - i])
    cells.append(cell)

lstm_cell = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)
value, _ = tf.nn.dynamic_rnn(lstm_cell, inputs, dtype=tf.float32)

weight = tf.Variable(tf.truncated_normal([lstmUnits, numClasses]))
bias = tf.Variable(tf.constant(0.1, shape=[numClasses]))
value = tf.transpose(value, [1, 0, 2])
last = tf.gather(value, int(value.get_shape()[0]) - 1)
prediction = tf.add(tf.matmul(last, weight), bias)

correctPred = tf.equal(tf.argmax(prediction, 1), tf.argmax(labels, 1))
accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32))

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=labels), name='loss')

# losses = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
#     [tf.reshape(prediction, [-1], name='reshape_pred')],
#     [tf.reshape(labels, [-1], name='reshape_target')],
#     [tf.ones([batch_size], dtype=tf.float32)],
#     average_across_timesteps=True,
#     softmax_loss_function=ms_error,
#     name='losses'
# )


# losses = tf.square(tf.subtract(labels, prediction))
#
# with tf.name_scope('average_cost'):
#     loss = tf.div(
#         tf.reduce_sum(losses, name='losses_sum'),
#         batch_size,
#         name='average_cost')
#     tf.summary.scalar('loss', loss)

global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)

learning_rate = tf.train.exponential_decay(0.005, global_step, 10000, 0.95, staircase=True)

train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss=loss, global_step=global_step)

tf.summary.scalar('Loss', loss)
tf.summary.scalar('Accuracy', accuracy)

merged = tf.summary.merge_all()
logdir = "tensorboard/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "/"

saver = tf.train.Saver()
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    writer = tf.summary.FileWriter(logdir, sess.graph)

    data = pd.read_csv('corpus_in_id.csv')
    data_count = 200000  # data.comment.count()
    for i in range(iterations):
        start = i * batch_size % data_count
        end = min(start + batch_size, data_count)
        # Next Batch of reviews
        nextBatch, nextBatchLabels = getBatch(data[start:end])

        _, acc, gs = sess.run([train_op, accuracy, global_step],
                              {input_data: nextBatch, labels: nextBatchLabels, keep_probs: [0.8, 0.7, 0.6, 0.5],
                               is_training: True})

        # Write summary to Tensorboard
        if i % 50 == 0:
            summary = sess.run(merged,
                               {input_data: nextBatch, labels: nextBatchLabels, keep_probs: [0.8, 0.7, 0.6, 0.5],
                                is_training: True})
            writer.add_summary(summary, i)

        if i % 1000 == 0:
            print('step {}: {}%'.format(gs, acc * 100))
            for j in range(10):
                start = j * batch_size % 200000
                end = min(start + batch_size, 200000)
                # Next Batch of reviews
                nextBatch, nextBatchLabels = getBatch(data[start:end])
                print("Accuracy for this batch:{}%".format((sess.run(accuracy,
                                                                     {input_data: nextBatch, labels: nextBatchLabels,
                                                                      keep_probs: [1, 1, 1, 1]})) * 100))

        # Save the network every 10,000 training iterations
        if (i % 50000 == 0 and i != 0):
            save_path = saver.save(sess, "models/pretrained_lstm.ckpt", global_step=i + 1)
            print("saved to %s" % save_path)
    writer.close()

    writer = tf.summary.FileWriter(logdir, sess.graph)
