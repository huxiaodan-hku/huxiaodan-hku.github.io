

```python
from __future__ import print_function, division
import numpy as np
import tensorflow as tf
```

# 定义参数


```python
num_epochs = 5   #全部数据一共反复训练次数
total_series_length = 50000   
truncated_backprop_length = 10     #输入X1,x2,...Xn 的长度
state_size = 4     #隐藏层大小
num_classes = 2     #输出的class ， 本程序只有0,1两种输出
echo_step = 3      #输出偏移位数
batch_size = 5     #batch_sizs
num_layers = 3     #多层隐藏层
num_batches = total_series_length//batch_size//truncated_backprop_length    #batch 的数量
```

# 生成数据
  
```
 inputs_value:          targets_value:
[[1 0 0 ..., 1 0 0]   [[0 0 0 ..., 1 1 0]
 [0 1 1 ..., 0 0 1]    [0 0 0 ..., 1 1 1]
 [0 1 0 ..., 1 1 1]    [0 0 0 ..., 0 0 0]
 [1 1 0 ..., 0 0 1]    [0 0 0 ..., 0 0 1]
 [0 1 1 ..., 1 0 0]]   [0 0 0 ..., 0 0 0]]
```


```python
inputs_value = np.array(np.random.choice(2, total_series_length, p=[0.5, 0.5]))
targets_value = np.roll(inputs_value, echo_step)
for i in range(num_batches * batch_size):
    targets_value[i * truncated_backprop_length : (i * truncated_backprop_length + echo_step)] = 0

inputs_value = inputs_value.reshape((batch_size, -1))  
targets_value = targets_value.reshape((batch_size, -1))
```

# 定义模型
```
rnn_outputs:
>>>Tensor("rnn/transpose:0", shape=(5, 10, 4), dtype=float32)
tf.reshape(rnn_outputs, [-1, state_size]):
>>>Tensor("Reshape:0", shape=(50, 4), dtype=float32)
```


```python
batchX_placeholder = tf.placeholder(tf.int32, [batch_size, truncated_backprop_length])
batchY_placeholder = tf.placeholder(tf.int32, [batch_size, truncated_backprop_length])

#one_hot_batchX_placeholder = tf.one_hot(batchX_placeholder,num_classes)
with tf.variable_scope("embedding",reuse=tf.AUTO_REUSE):
    embeddings = tf.get_variable('embedding_matrix', [num_classes, state_size])

rnn_inputs = tf.nn.embedding_lookup(embeddings, batchX_placeholder)

```


```python
lstm_cell = tf.contrib.rnn.BasicLSTMCell(state_size,state_is_tuple=True)
lstm_cell = tf.contrib.rnn.MultiRNNCell([lstm_cell] * num_layers, state_is_tuple=True)
init_state = lstm_cell.zero_state(batch_size, tf.float32)
rnn_outputs, final_state = tf.nn.dynamic_rnn(lstm_cell, rnn_inputs, initial_state=init_state)
```


```python
W2 = tf.Variable(np.random.rand(state_size, num_classes),dtype=tf.float32)
b2 = tf.Variable(np.zeros((1,num_classes)), dtype=tf.float32)

logits = tf.reshape(
            tf.matmul(tf.reshape(rnn_outputs, [-1, state_size]), W2) + b2,
            [batch_size, truncated_backprop_length, num_classes])

losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=batchY_placeholder, logits=logits)

total_loss = tf.reduce_mean(losses)
train_step = tf.train.AdagradOptimizer(0.3).minimize(total_loss)
```


```python
with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    loss_list = []

    for epoch_idx in range(num_epochs):
        _current_cell_state = np.zeros((batch_size, state_size))
        _current_hidden_state = np.zeros((batch_size, state_size))

        print("New data, epoch", epoch_idx)

        for batch_idx in range(num_batches):
            start_idx = batch_idx * truncated_backprop_length
            end_idx = start_idx + truncated_backprop_length

            batchX = inputs_value[:,start_idx:end_idx]
            batchY = targets_value[:,start_idx:end_idx]

            _total_loss, _train_step = sess.run(
                [total_loss, train_step],
                feed_dict={
                    batchX_placeholder: batchX,
                    batchY_placeholder: batchY,
                })

            loss_list.append(_total_loss)

            if batch_idx%100 == 0:
                print("Step",batch_idx, "Batch loss", _total_loss)
```

    WARNING:tensorflow:From D:\anaconda\envs\tensorflow\lib\site-packages\tensorflow\python\util\tf_should_use.py:107: initialize_all_variables (from tensorflow.python.ops.variables) is deprecated and will be removed after 2017-03-02.
    Instructions for updating:
    Use `tf.global_variables_initializer` instead.
    New data, epoch 0
    Step 0 Batch loss 0.693381
    Step 100 Batch loss 0.534515
    Step 200 Batch loss 0.484091
    Step 300 Batch loss 0.188863
    Step 400 Batch loss 0.0233409
    Step 500 Batch loss 0.0112662
    Step 600 Batch loss 0.0059204
    Step 700 Batch loss 0.00405966
    Step 800 Batch loss 0.00305155
    Step 900 Batch loss 0.00283688
    New data, epoch 1
    Step 0 Batch loss 0.00326661
    Step 100 Batch loss 0.00185313
    Step 200 Batch loss 0.00206363
    Step 300 Batch loss 0.00133301
    Step 400 Batch loss 0.00172968
    Step 500 Batch loss 0.0014067
    Step 600 Batch loss 0.00114917
    Step 700 Batch loss 0.000989825
    Step 800 Batch loss 0.000944063
    Step 900 Batch loss 0.00100075
    New data, epoch 2
    Step 0 Batch loss 0.00103392
    Step 100 Batch loss 0.00199549
    Step 200 Batch loss 0.00174496
    Step 300 Batch loss 0.000993447
    Step 400 Batch loss 0.00134069
    Step 500 Batch loss 0.00102726
    Step 600 Batch loss 0.000888835
    Step 700 Batch loss 0.000780438
    Step 800 Batch loss 0.000774152
    Step 900 Batch loss 0.000828426
    New data, epoch 3
    Step 0 Batch loss 0.000847003
    Step 100 Batch loss 0.000734727
    Step 200 Batch loss 0.000786956
    Step 300 Batch loss 0.000472508
    Step 400 Batch loss 0.000679311
    Step 500 Batch loss 0.000596685
    Step 600 Batch loss 0.000511092
    Step 700 Batch loss 0.000479914
    Step 800 Batch loss 0.000474668
    Step 900 Batch loss 0.000537077
    New data, epoch 4
    Step 0 Batch loss 0.000556837
    Step 100 Batch loss 0.000484772
    Step 200 Batch loss 0.00053663
    Step 300 Batch loss 0.000323339
    Step 400 Batch loss 0.00046523
    Step 500 Batch loss 0.00043071
    Step 600 Batch loss 0.000366404
    Step 700 Batch loss 0.000350811
    Step 800 Batch loss 0.000346866
    Step 900 Batch loss 0.000401205
    
