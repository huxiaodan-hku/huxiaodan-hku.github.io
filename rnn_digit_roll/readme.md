# introduction
利用了**数字转移**和**字符预测**两个项目，从底层的python实现，到使用高层tensorflow api，一步一步掌握tensorflow及RNN的使用。

# digit roll 数字转移
训练的输入是一个只有0和1的二进制数组，输出是将输入向右偏移固定位数，然后偏移过的部分用0补充。
例如下面的例子输出相对于输入偏移了3位。
input:  0 1 0 1 0 0 1 0 1 1
target: 0 0 0 0 1 0 1 0 0 1

详细内容如下：
[基于tensorflow1.4，使用rnn实现数字转移，从0开始系列](digit_roll_index.md)

# char predict

