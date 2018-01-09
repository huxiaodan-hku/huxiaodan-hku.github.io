# basic rnn model using tensorflow basic language
```python
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 13:47:26 2018

@author: xiaodan.hu
"""

import tensorflow as tf
import numpy as np

'''
initialization
'''
seq_length  = 10            # each input includes 10 digits
hidden_size = 5             # hidden layer has 5 neurons 
learning_rate = 0.3         # gradient descenting rate
batch_size = 1              # once processing number of inputs
roll_digit = 2              # target (roll from input)
num_batches = 1000          # train data size 
total_series_length = num_batches * batch_size * seq_length

'''
Generate training data
'''
inputs_value = np.array(np.random.choice(2, total_series_length, p=[0.5, 0.5]))
targets_value = np.roll(inputs_value, roll_digit)
for i in range(num_batches * batch_size):
    targets_value[i * seq_length : (i * seq_length + roll_digit)] = 0

inputs_value = inputs_value.reshape((batch_size, -1))  
targets_value = targets_value.reshape((batch_size, -1))

'''
build rnn model
'''
batch_input_placeholder = tf.placeholder(tf.float32, [batch_size, seq_length])
batch_target_placeholder = tf.placeholder(tf.int32, [batch_size, seq_length])


W = tf.Variable(np.random.rand(hidden_size+1, hidden_size), dtype=tf.float32)
b = tf.Variable(np.zeros((1,hidden_size)), dtype=tf.float32)

W2 = tf.Variable(np.random.rand(hidden_size, num_classes),dtype=tf.float32)
b2 = tf.Variable(np.zeros((1,num_classes)), dtype=tf.float32) 

inputs_series = tf.unstack(batch_input_placeholder, axis=1)
labels_series = tf.unstack(batch_target_placeholder, axis=1)

init_state = tf.placeholder(tf.float32, [batch_size, hidden_size])

current_state = init_state
states_series = []
for current_input in inputs_series: 
    current_input = tf.reshape(current_input, [batch_size, 1])
    input_and_state_concatenated = tf.concat([current_input, current_state], 1)
    next_state = tf.tanh(tf.matmul(input_and_state_concatenated, W) + b)  # Broadcasted addition
    states_series.append(next_state)
    current_state = next_state


logits_series = [tf.matmul(state, W2) + b2 for state in states_series] #Broadcasted addition
predictions_series = [tf.nn.softmax(logits) for logits in logits_series]
losses = [tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels) for logits, labels in zip(logits_series,labels_series)]
total_loss = tf.reduce_mean(losses)
train_step = tf.train.AdagradOptimizer(learning_rate).minimize(total_loss)

'''
write relative data to file
'''


input_txt = []
target_txt = []
weight_H_H_txt = []
weight_X_H_txt = []
weight_H_Y_txt = []
weight_b_H_txt = []
weight_b_Y_txt = [] 
loss_list = []

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for index in range(num_batches):
        _current_state = np.zeros((batch_size, hidden_size))
        start_idx = index * seq_length
        end_idx = start_idx + seq_length
  
        batchX = inputs_value[:,start_idx:end_idx]       
        batchY = targets_value[:,start_idx:end_idx]
        
        _total_loss, _train_step, _current_state, _predictions_series, _states_series = sess.run(
            [total_loss, train_step, current_state, predictions_series, states_series],
            feed_dict={
                batch_input_placeholder:batchX,
                batch_target_placeholder:batchY,
                init_state:_current_state
            })
        
    
        whole_state_series.append(_states_series)
        loss_list.append(_total_loss) #lost
        input_txt.append(batchX) #input
        target_txt.append(batchY) #output
        weight_H_H_txt.append(W.eval()[0:hidden_size,:])
        weight_X_H_txt.append(W.eval()[hidden_size:hidden_size+1,:])
        weight_H_Y_txt.append(W2.eval())
        weight_b_H_txt.append(b.eval()) 
        weight_b_Y_txt.append(b2.eval())     
        print("Step",index, "Loss", _total_loss)
        


def write_to_file(data,file):
    with open(file,'a',encoding='utf-8') as f:
        for temp in data:
            for index,each_batch in enumerate(temp):
                for single_data in each_batch:
                    f.write(str(single_data)+" ")
                f.write('\n')
                f.write("batch:%d\n"%index)
                f.write('\n')
            
            
        
filename="shiyishi.txt"

write_to_file([input_txt,target_txt,weight_X_H_txt,weight_H_H_txt,weight_H_Y_txt,weight_b_H_txt,weight_b_Y_txt,whole_state_series],filename)
with open(filename,'a',encoding='utf-8') as f:
    for temp in loss_list:
        f.write(str(temp))
        f.write('\n')

                
```
