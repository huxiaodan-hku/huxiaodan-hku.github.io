# digit roll
利用了**数字转移**项目，从底层的python实现，到使用高层tensorflow api，一步一步掌握tensorflow及RNN的使用。

# 训练数据
训练的输入是一个只有0和1的二进制数组，输出是将输入向右偏移固定位数，然后偏移过的部分用0补充。
例如下面的例子输出相对于输入偏移了3位。
input:  0 1 0 1 0 0 1 0 1 1
target: 0 0 0 0 1 0 1 0 0 1

# 内容
[基于tensorflow1.4，使用rnn实现数字转移，从0开始系列一](rnn_basic.md)

[基于tensorflow1.4，使用rnn实现数字转移，从0开始系列二](rnn_basic_tensorflow.md)

[基于tensorflow1.4，使用rnn实现数字转移，从0开始系列三](rnn_basic_cell.md)

[基于tensorflow1.4，使用rnn实现数字转移，从0开始系列四](rnn_lstm_cell.md)

[基于tensorflow1.4，使用rnn实现数字转移，从0开始系列五](rnn_lstm_mutiple_cell.md)

