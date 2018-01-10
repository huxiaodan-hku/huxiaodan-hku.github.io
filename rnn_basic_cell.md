#使用tensorflow Rnn_cell api 实现
tensorflow 1.4
python3+


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


cell = tf.contrib.rnn.BasicRNNCell(state_size)
init_state = tf.zeros([batch_size, state_size])

rnn_outputs, final_state = tf.nn.dynamic_rnn(cell, one_hot_batchX_placeholder, initial_state=init_state)


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
    Step 0 Batch loss 0.70098
    Step 100 Batch loss 0.42059
    Step 200 Batch loss 0.010894
    Step 300 Batch loss 0.00570092
    Step 400 Batch loss 0.00321913
    Step 500 Batch loss 0.00323861
    Step 600 Batch loss 0.00242459
    Step 700 Batch loss 0.00200511
    Step 800 Batch loss 0.00172231
    Step 900 Batch loss 0.00124845
    New data, epoch 1
    Step 0 Batch loss 0.00105245
    Step 100 Batch loss 0.00121114
    Step 200 Batch loss 0.000874155
    Step 300 Batch loss 0.000854327
    Step 400 Batch loss 0.000702099
    Step 500 Batch loss 0.000879682
    Step 600 Batch loss 0.000777922
    Step 700 Batch loss 0.000725159
    Step 800 Batch loss 0.000692487
    Step 900 Batch loss 0.000545684
    New data, epoch 2
    Step 0 Batch loss 0.000490697
    Step 100 Batch loss 0.000592507
    Step 200 Batch loss 0.00044976
    Step 300 Batch loss 0.000456264
    Step 400 Batch loss 0.000391093
    Step 500 Batch loss 0.000504155
    Step 600 Batch loss 0.000460278
    Step 700 Batch loss 0.00043969
    Step 800 Batch loss 0.000430921
    Step 900 Batch loss 0.000348337
    New data, epoch 3
    Step 0 Batch loss 0.000319283
    Step 100 Batch loss 0.000390936
    Step 200 Batch loss 0.000301999
    Step 300 Batch loss 0.000310358
    Step 400 Batch loss 0.00027047
    Step 500 Batch loss 0.000352204
    Step 600 Batch loss 0.000326097
    Step 700 Batch loss 0.000314724
    Step 800 Batch loss 0.000312079
    Step 900 Batch loss 0.000255602
    New data, epoch 4
    Step 0 Batch loss 0.000236439
    Step 100 Batch loss 0.000291273
    Step 200 Batch loss 0.000227083
    Step 300 Batch loss 0.000234835
    Step 400 Batch loss 0.000206511
    Step 500 Batch loss 0.0002702
    Step 600 Batch loss 0.000252213
    Step 700 Batch loss 0.00024475
    Step 800 Batch loss 0.000244336
    Step 900 Batch loss 0.000201769
    
