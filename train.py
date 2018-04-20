#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import logging
import os

import tensorflow as tf

import utils
from model import Model
from utils import read_data
from utils import index_data
import numpy as np

from flags import parse_args
FLAGS, unparsed = parse_args()


logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s', level=logging.DEBUG)


vocabulary,num_classes = read_data(FLAGS.text)

print('Data size', len(vocabulary))
print('num_classes', num_classes)

with open(FLAGS.dictionary, encoding='utf-8') as inf:
    dictionary = json.load(inf, encoding='utf-8')

with open(FLAGS.reverse_dictionary, encoding='utf-8') as inf:
    reverse_dictionary = json.load(inf, encoding='utf-8')

raw_x = index_data(vocabulary, dictionary)
raw_y = index_data(vocabulary[1:], dictionary)
raw_y[len(raw_y)-1] = num_classes-1

model = Model(learning_rate=FLAGS.learning_rate, batch_size=FLAGS.batch_size, num_steps=FLAGS.num_steps)
model.build(FLAGS.embedding_file)

training_state = None
with tf.Session() as sess:
    summary_string_writer = tf.summary.FileWriter(FLAGS.output_dir, sess.graph)

    saver = tf.train.Saver(max_to_keep=5)
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    logging.debug('Initialized')

    try:
        checkpoint_path = tf.train.latest_checkpoint(FLAGS.output_dir)
        saver.restore(sess, checkpoint_path)
        logging.debug('restore from [{0}]'.format(checkpoint_path))

    except Exception:
        logging.debug('no check point found....')

    state = sess.run(model.init_state)
    for epoch in range(10):
        logging.debug('epoch [{0}]....'.format(epoch))

        #for dl in utils.get_train_data(vocabulary, batch_size=FLAGS.batch_size, num_steps=FLAGS.num_steps):
        for step, (x,y) in enumerate(utils.get_train_data(len(vocabulary),raw_x,raw_y,batch_size=FLAGS.batch_size, num_steps=FLAGS.num_steps)):

            ##################
            # Your Code here
            ##################
            feed_dict = {model.X:x, model.Y:y,model.keep_prob:0.5,model.init_state:state}

            '''gs, _, state, l, summary_string = sess.run(
                [model.global_step, model.optimizer, model.outputs_state_tensor, model.loss, model.merged_summary_op], feed_dict=feed_dict)'''
            gs, _, state, l, summary_string = sess.run(
                [model.global_step, model.optimizer, model.final_state, model.loss, model.merged_summary_op], feed_dict=feed_dict)
            summary_string_writer.add_summary(summary_string, gs)

            if gs % 10 == 0:
                logging.debug('step [{0}] loss [{1}]'.format(gs, l))
                save_path = saver.save(sess, os.path.join(
                    FLAGS.output_dir, "model.ckpt"), global_step=gs)
    summary_string_writer.close()
