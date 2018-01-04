# python rnn
```
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 13:47:26 2018

@author: xiaodan.hu
"""

import tensorflow as tf
import numpy as np

# 训练集合为10000个
train_size = 10
seq_length  = 10
test_size = 10
hidden_size = 10
learning_rate = 0.1
vocab_size = 2
batch_size =5


inputs_value = np.random.randint(2, size=(train_size, seq_length, 1))
inputs = tf.placeholder(tf.float32,[train_size,seq_length,1])
targets = tf.placeholder(tf.float32,[train_size,seq_length,1])

init_state = tf.placeholder(tf.float32, [hidden_size,1])
current_state = init_state
state_series = []

for row_indice in tf.range(train_size):
    print(row_indice)
with tf.Session() as sess:
    row_indices  = [0]
    row = tf.gather(inputs,row_indices)
    print(sess.run(row,feed_dict={inputs:inputs_value}))
```
