# tf.squeeze
```python
squeeze(
    input,
    axis=None,
    name=None,
    squeeze_dims=None
)
```
这个函数的作用是把只有1个元素的维度给取消。
```python
# 't' is a tensor of shape [1, 2, 1, 3, 1, 1]
tf.shape(tf.squeeze(t))  # [2, 3]
```
还可以专门指定取消哪几个维度。
```python
# 't' is a tensor of shape [1, 2, 1, 3, 1, 1]
tf.shape(tf.squeeze(t, [2, 4]))  # [1, 2, 3, 1]
```
