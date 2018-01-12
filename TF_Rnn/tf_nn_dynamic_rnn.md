```python
dynamic_rnn(
    cell, #所有的例如lstm神经元
    inputs, 输入的序列（embedding之后的）
    sequence_length=None, 传入一个向量[batch_size]
    initial_state=None, 所有神经元的初始状态
    dtype=None,
    parallel_iterations=None,
    swap_memory=False,
    time_major=False,
    scope=None
)
```
initial state 可以用下列方法
```python
encoder_cell = tf.contrib.rnn.BasicLSTMCell(encoder_hidden_size,state_is_tuple=True)
encoder_cell = tf.contrib.rnn.DropoutWrapper(encoder_cell, output_keep_prob=0.8)
encoder_cell = tf.contrib.rnn.MultiRNNCell([encoder_cell] * num_layers, state_is_tuple=True)

init_state = encoder_cell.zero_state(batch_size, tf.float32)
```
