# tensorflow Variable

创建一个Variable，这里默认是tf.float32类型的，默认的初始值是使用`tf.glorot_unifor_initializer`来初始化的
```python
my_variable = tf.get_variable("my_variable", [1, 2, 3])
```
自己指定Variable的类型和初始方式。
```python
my_int_variable = tf.get_variable("my_int_variable", [1, 2, 3], dtype=tf.int32, 
  initializer=tf.zeros_initializer)
```
利用一个tf.Tensor来创建变量。
```python
other_variable = tf.get_variable("other_variable", dtype=tf.int32, 
  initializer=tf.constant([23, 42]))
```

### Collection
tensorflow中的变量有三种类型，也称之为：
- `tf.GraphKeys.GLOBAL_VARIABLES`：这种变量可以被多种设备共享
- `tf.GraphKeys.TRAINABLE_VARIABLES`：这种变量主要用于训练使用，比如计算梯度
- `tf.GraphKeys.LOCAL_VARIABLES`：不可训练用。

将一个变量添加到collection的方法是：
```python
my_local = tf.get_variable("my_local", shape=(), 
collections=[tf.GraphKeys.LOCAL_VARIABLES])
```
或者使用
```python
my_non_trainable = tf.get_variable("my_non_trainable", 
                                   shape=(), 
                                   trainable=False)
```
也可以自定义collection
```python
tf.add_to_collection("my_collection_name", my_local)

#retrieve a list of all the variables 
tf.get_collection("my_collection_name") 
```

### 初始化变量
变量在使用前需要进行初始化，一次性初始化所有变量的方法是`tf.global_variables_initializer()`
它会把`tf.GraphKeys.GLOBAL_VARIABLES`中所有的变量都初始化。
```python
session.run(tf.global_variables_initializer())
# Now all variables are initialized.
```
如果想查看还有哪些变量未被初始化，可以使用以下命令
```python
print(session.run(tf.report_uninitialized_variables()))
```
由于`tf.global_variables_initializer`在执行的时候并未考虑变量初始的顺序，而有些变量的初始是依靠其他变量的。
这时如果初始顺序错误，程序就会报错。所以我们最好使用`variable.initialized_value()`来主动初始化。
```python
v = tf.get_variable("v", shape=(), initializer=tf.zeros_initializer())
w = tf.get_variable("w", initializer=v.initialized_value() + 1)
```

### 使用变量
可以把变量看成tensor直接进行运算
```python
v = tf.get_variable("v", shape=(), initializer=tf.zeros_initializer())
w = v + 1  # w is a tf.Tensor which is computed based on the value of v.
           # Any time a variable is used in an expression it gets automatically
           # converted to a tf.Tensor representing its value.
```
给变量重新赋值可以使用`assign`,`assing_add`等方法
```python
v = tf.get_variable("v", shape=(), initializer=tf.zeros_initializer())
assignment = v.assign_add(1)
tf.global_variables_initializer().run()
assignment.run()
```
很多时候，我们回归模型的过程中，都需要不断更新variable的值。我们需要知道当前variable的值是属于哪个时间点的。
我们可以使用`tf.Variable.read_value。例如：
```python
v = tf.get_variable("v", shape=(), initializer=tf.zeros_initializer())
assignment = v.assign_add(1)
with tf.control_dependencies([assignment]):
  w = v.read_value()  # w is guaranteed to reflect v's value after the
                      # assign_add operation.
```    

### 共享变量
比如在某一步运算中我们使用weight和bias变量做一些卷积运算
```python
def conv_relu(input, kernel_shape, bias_shape):
    # Create variable named "weights".
    weights = tf.get_variable("weights", kernel_shape,
        initializer=tf.random_normal_initializer())
    # Create variable named "biases".
    biases = tf.get_variable("biases", bias_shape,
        initializer=tf.constant_initializer(0.0))
    conv = tf.nn.conv2d(input, weights,
        strides=[1, 1, 1, 1], padding='SAME')
    return tf.nn.relu(conv + biases)
```
但是，当我们有很多卷积层，反复调用这个方法就会出现问题。
```python
input1 = tf.random_normal([1,10,10,32])
input2 = tf.random_normal([1,20,20,32])
x = conv_relu(input1, kernel_shape=[5, 5, 32, 32], bias_shape=[32])
x = conv_relu(x, kernel_shape=[5, 5, 32, 32], bias_shape = [32])  # This fails.
```
解决的方法就是使用`variable_scope`，设定不同命名空间
```python
def my_image_filter(input_images):
    with tf.variable_scope("conv1"):
        # Variables created here will be named "conv1/weights", "conv1/biases".
        relu1 = conv_relu(input_images, [5, 5, 32, 32], [32])
    with tf.variable_scope("conv2"):
        # Variables created here will be named "conv2/weights", "conv2/biases".
        return conv_relu(relu1, [5, 5, 32, 32], [32])
```
当想反复使用同一个命名空间时，我们可以指定该变量可以*reuse*
```python
with tf.variable_scope("model"):
  output1 = my_image_filter(input1)
with tf.variable_scope("model", reuse=True):
  output2 = my_image_filter(input2)
```
或者使用`scope.reuse_variable()`
```python
with tf.variable_scope("model") as scope:
  output1 = my_image_filter(input1)
  scope.reuse_variables()
  output2 = my_image_filter(input2)

def foo():
  with tf.variable_scope("foo", reuse=tf.AUTO_REUSE):
    v = tf.get_variable("v", [1])
  return v
```
