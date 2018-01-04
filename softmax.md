[tensorflow 基础]()

#*MINIST原理*
假设现在有1个10*10小格的像素图如下
![](http://upload-images.jianshu.io/upload_images/4037309-9fc2bb4abe0ac352.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
假设每张图都代表0-9中的一个数字，如果有一张新图，我们想知道他是数字几该怎么判断？
![](http://upload-images.jianshu.io/upload_images/4037309-2ad802f4775e709f.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

我们知道MNIST的每一张图片都表示一个数字，从0到9。我们希望得到给定图片代表每个数字的概率。比如说，我们的模型可能推测一张包含9的图片代表数字9的概率是80%但是判断它是8的概率是5%（因为8和9都有上半部分的小圆），然后给予它代表其他数字的概率更小的值。

为了求A是各个数字相似的概率，我们可以给A与数字0-9的相似度打一个分，与一个图片相似度越高，这个分数越高，从而概率也就越高。

我们称这个分数为evidence，为了得到一张给定图片属于某个特定数字类的证据（evidence），我们对图片像素值进行加权求和。如果这个像素具有很强的证据说明这张图片不属于该类，那么相应的权值为负数，相反如果这个像素拥有有利的证据支持这张图片属于这个类，那么权值是正数。
[图片上传失败...(image-2ddf5c-1512980768026)]
我们也需要加入一个额外的偏置量（bias），因为输入往往会带有一些无关的干扰量。因此对于给定的输入图片 x 它代表的是数字 i 的证据可以表示为
![](http://upload-images.jianshu.io/upload_images/4037309-b3c7291be515dab3.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

其中i代表数字索引，j代表像素索引。


这下，我们就可以得到给定图片代表每个数字的分数evidence，那么如何这个分数转化为更为直观的概率呢？

#*softmax模型*
这里我们要使用一种叫作softmax回归（softmax regression）的模型。softmax模型可以用来给不同的对象分配概率。也可以将softmax看成是一个激励（activation）函数或者链接（link）函数，把我们定义的线性函数的输出转换成我们想要的格式，也就是关于10个数字类的概率分布。因此，给定一张图片，它对于每一个数字的吻合度可以被softmax函数转换成为一个概率值。softmax函数可以定义为：
![](http://upload-images.jianshu.io/upload_images/4037309-157d459dea736bd8.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

把上图等式右边带入公式可得：
![](http://upload-images.jianshu.io/upload_images/4037309-0ab382054cae6106.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

但是更多的时候把softmax模型函数定义为前一种形式：把输入值当成幂指数求值，再正则化这些结果值。这个幂运算表示，更大的证据对应更大的假设模型（hypothesis）里面的乘数权重值。反之，拥有更少的证据意味着在假设模型里面拥有更小的乘数系数。假设模型里的权值不可以是0值或者负值。Softmax然后会正则化这些权重值，使它们的总和等于1，以此构造一个有效的概率分布。

对于softmax回归模型可以用下面的图解释，对于输入的`xs`加权求和，再分别加上一个偏置量，最后再输入到softmax函数中：

![](http://upload-images.jianshu.io/upload_images/4037309-7ce8e9e48c211c54.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
如果把它写成一个等式，我们可以得到：
[图片上传失败...(image-1922ac-1512980768026)]

我们也可以用向量表示这个计算过程：用矩阵乘法和向量相加。这有助于提高计算效率。（也是一种更有效的思考方式）
![](http://upload-images.jianshu.io/upload_images/4037309-760f84e63e94ab0b.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

更进一步，可以写成更加紧凑的方式：
![](http://upload-images.jianshu.io/upload_images/4037309-f299c64810157797.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

好了，模型建好了，为了训练我们的模型，我们首先需要定义一个指标来评估这个模型是好的。其实，在机器学习，我们通常定义指标来表示一个模型是坏的，这个指标称为成本（cost）或损失（loss），然后尽量最小化这个指标。但是，这两种方式是相同的。

一个非常常见的，非常漂亮的成本函数是“交叉熵”（cross-entropy）。下面我们来解释一下什么是交叉熵。

#熵 entropy

小明发明了一种通讯编码方式，他将“狗”，“猫”，“鱼”，“鸟”几种动物用一个2bit的2进制来进行编码。编码如下图：
![](http://upload-images.jianshu.io/upload_images/4037309-f6202692dafd78fd.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

由于每个bit的传输价格非常昂贵，小明希望能够优化编码，能用最短的bit位数就能区分这四个动物。于是天才胡老大同学接下了这个任务，他发现，小明使用这四个单词的概率分如下图：

![](http://upload-images.jianshu.io/upload_images/4037309-53ae8d65eba9f165.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


如果按小明的方式编码，那么每个单词的期望编码位数为：
**1/2 * 2 + 1/4 * 2 + 1/8 * 2 + 1/8 * 2 = 2 bits**

胡老大想出了一种更高效的编码

![](http://upload-images.jianshu.io/upload_images/4037309-a350834f274bd7a6.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

其中每个单词的位数计算公式如下
![](http://upload-images.jianshu.io/upload_images/4037309-0f0ffda133855b8e.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


所以单个单词的期望编码位数为：
**1/2 * 1 + 1/4 * 2 + 1/8 * 3 + 1/8 * 3 = 7/4 = 1.75 bits.**

明显胡老大识别每个单词期望的位数1.75小于小明识别每个单词所期望的位数。我们称这种识别信息所需的最小位数称为 **entropy**，也就是 **熵**。**H(p) = ∑p(x) * log2(1/p(x))** 或者 **H(p) = −∑p(x)log2(p(x))**

#**Cross-entropy**
假设小丽是一个猫爱好者，她和小明的信息传输概率分布如下：
[图片上传失败...(image-800784-1512981095761)]
小丽比较懒，她直接使用用了小明的编码方式，最后她的最小位数为：
**1 * 1/8+2 * 1/2+3 * 1/4+3 * 1/4 = 17/8 = 2.25**
明显高于1.75!!!!
那么这个使用小明最优编码求出来的2.25称之为cross-entropy
**Hp(q)=∑q(x) * log2(1/p(x))**

小丽使用自己的编码： (H(q)=1.75 bits)
小丽使用小明的编码： (Hp(q)=2.25 bits)
小明使用自己的编码： (H(p)=1.75 bits)
小明使用小丽的编码： (Hq(p)=2.375 bits)

可以发现Hp(q) 不等于 Hq(p)，所以cross-entropy是非对称的。cross-entropy还可以帮我们分辨两个事件概率分布的差异。两个事件概率分布差异越大，cross-entropy也就越大。

#**KL divergence**
相比于cross-entropy，我们对cross-entropy与entropy之间的差值更感兴趣。我们称这个差值为Kullback–Leibler divergence，**Dq(p)=Hq(p)−H(p)**，KL divergence就好像两个分布之间的距离，用来测量两个分布的不同程度。

#**训练模型**
重新回到我们的训练模型上。
[图片上传失败...(image-2f704c-1512981095761)]
y 是我们预测的概率分布, y' 是实际的分布（我们输入的one-hot vector)。

为了计算交叉熵，我们首先需要添加一个新的占位符用于输入正确值：

```
y_ = tf.placeholder("float", [None,10])
```
然后我们可以用 计算交叉熵:
```
cross_entropy = -tf.reduce_sum(y_*tf.log(y))
```
我们的训练目的就是使cross-entropy的值越小越好，因为TensorFlow拥有一张描述你各个计算单元的图，它可以自动地使用[backpropagation algorithm](http://colah.github.io/posts/2015-08-Backprop/)反向传播算法。来有效地确定你的变量是如何影响你想要最小化的那个成本值的。然后，TensorFlow会用你选择的优化算法来不断地修改变量以降低成本。
```
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
```
在这里，我们要求TensorFlow用梯度下降算法（gradient descent algorithm）以0.01的学习速率最小化交叉熵。梯度下降算法（gradient descent algorithm）是一个简单的学习过程，TensorFlow只需将每个变量一点点地往使成本不断降低的方向移动。当然TensorFlow也提供了[其他许多优化算法]

TensorFlow在这里实际上所做的是，它会在后台给描述你的计算的那张图里面增加一系列新的计算操作单元用于实现反向传播算法和梯度下降算法。然后，它返回给你的只是一个单一的操作，当运行这个操作时，它用梯度下降算法训练你的模型，微调你的变量，不断减少成本。

现在，我们已经设置好了我们的模型。在运行计算之前，我们需要添加一个操作来初始化我们创建的变量：
```
init = tf.initialize_all_variables()
```
现在我们可以在一个Session里面启动我们的模型，并且初始化变量：
```
sess = tf.Session()
sess.run(init)
```
然后开始训练模型，这里我们让模型循环训练1000次！
```
for i in range(1000):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
```
该循环的每个步骤中，我们都会随机抓取训练数据中的100个批处理数据点，然后我们用这些数据点作为参数替换之前的占位符来运行train_step。

使用一小部分的随机数据来进行训练被称为随机训练（stochastic training）- 在这里更确切的说是随机梯度下降训练。在理想情况下，我们希望用我们所有的数据来进行每一步的训练，因为这能给我们更好的训练结果，但显然这需要很大的计算开销。所以，每一次训练我们可以使用不同的数据子集，这样做既可以减少计算开销，又可以最大化地学习到数据集的总体特性。

## 评估模型

那么我们的模型性能如何呢？

首先让我们找出那些预测正确的标签。来检测我们的预测是否真实标签匹配(索引位置一样表示匹配)。

```
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
```

这行代码会给我们一组布尔值。为了确定正确预测项的比例，我们可以把布尔值转换成浮点数，然后取平均值。例如，`[True, False, True, True]`会变成`[1,0,1,1]`，取平均值后得到`0.75`.
```
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
```

最后，我们计算所学习到的模型在测试数据集上面的正确率。

```
print sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})
```
这个最终结果值应该大约是91%。


