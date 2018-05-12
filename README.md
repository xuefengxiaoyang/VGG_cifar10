# VGG_cifar10
Classifying cifar-10 data sets using VGG networks

这三个python文件实现了用VGG16网络对cifar-10数据集进行分类，dataLoad.py文件负责读取二进制数据数据，直接运行可以将图片展示出来，也可以保存，在当前根目录
新建一个文件夹保存即可。train.py文件负责训练网络，VGG.py文件包含了网络的主要架构。因为电脑显卡不行，耗时太久，只训练了2000张图片，一个epoch还不到，训练
集总共50000张图。初始参数分类精度在10%左右，提高到了50%多，若继续训练，精度还会提高。
