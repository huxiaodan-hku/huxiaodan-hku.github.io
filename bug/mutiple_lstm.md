### 解决之前的代码

```python
init_state = tf.placeholder(tf.float32, [num_layers, 2, batch_size, state_size])
state_per_layer_list = tf.unstack(init_state, axis=0)
rnn_tuple_state = tuple(
    [tf.nn.rnn_cell.LSTMStateTuple(state_per_layer_list[idx][0], state_per_layer_list[idx][1])
     for idx in range(num_layers)]
)

_current_state = np.zeros((num_layers, 2, batch_size, state_size))
# Forward passes
cell = tf.contrib.rnn.BasicLSTMCell(state_size, state_is_tuple=True)
cell = tf.contrib.rnn.MultiRNNCell([cell] * num_layers, state_is_tuple=True)
states_series, current_state = tf.nn.dynamic_rnn(cell, one_hot_batchX_placeholder, initial_state=rnn_tuple_state)
```
### 报错信息
>ValueError: Variable rnn/basic_lstm_cell/kernel already exists, disallowed. Did you mean to set reuse=True or reuse=tf.AUTO_REUSE in VarScope? Originally defined at:

>File "D:\anaconda\envs\tensorflow\lib\site-packages\tensorflow\python\framework\ops.py", line 1470, in __init__self._traceback = self._graph._extract_stack()  # pylint: disable=protected-access
File "D:\anaconda\envs\tensorflow\lib\site-packages\tensorflow\python\framework\ops.py", line 2956, in create_op op_def=op_def)
File "D:\anaconda\envs\tensorflow\lib\site-packages\tensorflow\python\framework\op_def_library.py", line 787, in _apply_op_helper
op_def=op_def)
    
### 解决以后的代码
```python
batchX_placeholder = tf.placeholder(tf.int32, [batch_size, truncated_backprop_length])
batchY_placeholder = tf.placeholder(tf.int32, [batch_size, truncated_backprop_length])

#one_hot_batchX_placeholder = tf.one_hot(batchX_placeholder,num_classes)
with tf.variable_scope("embedding",reuse=tf.AUTO_REUSE):
    embeddings = tf.get_variable('embedding_matrix', [num_classes, state_size])

rnn_inputs = tf.nn.embedding_lookup(embeddings, batchX_placeholder)

lstm_cell = tf.contrib.rnn.BasicLSTMCell(state_size,state_is_tuple=True)
lstm_cell = tf.contrib.rnn.MultiRNNCell([lstm_cell] * num_layers, state_is_tuple=True)
init_state = lstm_cell.zero_state(batch_size, tf.float32)
rnn_outputs, final_state = tf.nn.dynamic_rnn(lstm_cell, rnn_inputs, initial_state=init_state)
```
