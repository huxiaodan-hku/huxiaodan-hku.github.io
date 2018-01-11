# tf.split
```python
split(
    value,
    num_or_size_splits,
    axis=0,
    num=None,
    name='split'
)
```
##### 功能
讲一个tensor分割成多个tensor
如果**num_or_size_splits**是一个整数类型，就表明要分割以后的tensor的数量。
如果**num_or_size_splits**是一个数组类型，就按数组要求分割。

##### 举例
```python
# 'value' is a tensor with shape [5, 30]
# Split 'value' into 3 tensors with sizes [4, 15, 11] along dimension 1
split0, split1, split2 = tf.split(value, [4, 15, 11], 1)
tf.shape(split0)  # [5, 4]
tf.shape(split1)  # [5, 15]
tf.shape(split2)  # [5, 11]
# Split 'value' into 3 tensors along dimension 1
split0, split1, split2 = tf.split(value, num_or_size_splits=3, axis=1)
tf.shape(split0)  # [5, 10]
```
