# - * -coding: utf-8

import tensorflow as tf

checkpoint_dir = './model'


with tf.Session() as session:
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        saver = tf.train.import_meta_graph(''.join([ckpt.model_checkpoint_path, '.meta']))
        saver.restore(session, ckpt.model_checkpoint_path)
    print(session.run('normed_embedding:0').shape)
