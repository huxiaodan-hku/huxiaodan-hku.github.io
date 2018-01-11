embedding_lookup(
    params,
    ids,
    partition_strategy='mod',
    name=None,
    validate_indices=True,
    max_norm=None
)


```python
num_classs = 2
state_size = 4
batch_size = 5
num_steps = 10
x = tf.placeholder(tf.int32, [batch_size, num_steps], name='input_placeholder')

embeddings = tf.get_variable('embedding_matrix', [num_classes, state_size])
tmp1 = tf.nn.embedding_lookup(embeddings, x)
# tmp1的值为：Tensor("embedding_lookup:0", shape=(5, 10, 4), dtype=float32)
```

x 的 值如下
```
[[1 0 0 1 0 0 1 1 0 0] 
 [0 1 1 1 0 0 1 0 0 1] 
 [0 1 0 1 0 0 1 1 1 1] 
 [1 1 0 1 0 0 1 0 0 1] 
 [0 1 1 1 0 0 1 1 0 0]]
```
embeddings 的值如下
```
[[ 0.37562275  0.7833004   0.45156384 -0.85742235]
 [ 0.35939527 -0.24664903 -0.37774205 -0.75642872]]
```
这里函数的意义就是拿embeddings的第一行替换上文中的输入0，拿embeddings的第二行替换上文中的1.
