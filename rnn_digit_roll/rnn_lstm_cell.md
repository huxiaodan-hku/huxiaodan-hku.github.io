

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

one_hot_batchX_placeholder = tf.one_hot(batchX_placeholder , num_classes)

lstm_cell = tf.contrib.rnn.BasicLSTMCell(state_size,state_is_tuple=True)
init_state = lstm_cell.zero_state(batch_size, tf.float32)



```


```python
rnn_outputs, final_state = tf.nn.dynamic_rnn(lstm_cell, one_hot_batchX_placeholder, initial_state=init_state)

W2 = tf.Variable(np.random.rand(state_size, num_classes),dtype=tf.float32)
b2 = tf.Variable(np.zeros((1,num_classes)), dtype=tf.float32)

logits = tf.reshape(
            tf.matmul(tf.reshape(rnn_outputs, [-1, state_size]), W2) + b2,
            [batch_size, truncated_backprop_length, num_classes])

losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=batchY_placeholder, logits=logits)

total_loss = tf.reduce_mean(losses)
train_step = tf.train.AdagradOptimizer(0.3).minimize(total_loss)
```

# 训练模型


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
    Step 0 Batch loss 0.665342
    Step 100 Batch loss 0.497867
    Step 200 Batch loss 0.250234
    Step 300 Batch loss 0.0182744
    Step 400 Batch loss 0.00721572
    Step 500 Batch loss 0.00526691
    Step 600 Batch loss 0.00396986
    Step 700 Batch loss 0.00424801
    Step 800 Batch loss 0.00280157
    Step 900 Batch loss 0.00229275
    New data, epoch 1
    Step 0 Batch loss 0.00187406
    Step 100 Batch loss 0.00195644
    Step 200 Batch loss 0.00161599
    Step 300 Batch loss 0.00121237
    Step 400 Batch loss 0.000907621
    Step 500 Batch loss 0.00102082
    Step 600 Batch loss 0.000997168
    Step 700 Batch loss 0.0012398
    Step 800 Batch loss 0.000896977
    Step 900 Batch loss 0.000827482
    New data, epoch 2
    Step 0 Batch loss 0.000701873
    Step 100 Batch loss 0.000871613
    Step 200 Batch loss 0.000734697
    Step 300 Batch loss 0.000591609
    Step 400 Batch loss 0.000459447
    Step 500 Batch loss 0.000540652
    Step 600 Batch loss 0.000552141
    Step 700 Batch loss 0.000716457
    Step 800 Batch loss 0.000522385
    Step 900 Batch loss 0.000499511
    New data, epoch 3
    Step 0 Batch loss 0.000423569
    Step 100 Batch loss 0.000555285
    Step 200 Batch loss 0.000467098
    Step 300 Batch loss 0.000383811
    Step 400 Batch loss 0.000306482
    Step 500 Batch loss 0.000364281
    Step 600 Batch loss 0.000382478
    Step 700 Batch loss 0.000499499
    Step 800 Batch loss 0.00036612
    Step 900 Batch loss 0.000354999
    New data, epoch 4
    Step 0 Batch loss 0.000301927
    Step 100 Batch loss 0.000404567
    Step 200 Batch loss 0.000339338
    Step 300 Batch loss 0.00028178
    Step 400 Batch loss 0.0002296
    Step 500 Batch loss 0.000273609
    Step 600 Batch loss 0.000292192
    Step 700 Batch loss 0.000381958
    Step 800 Batch loss 0.000281514
    Step 900 Batch loss 0.000274644
    
