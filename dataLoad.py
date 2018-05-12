import tensorflow as tf
import matplotlib.pyplot  as plt
from PIL import Image

dir = "cifar-10-batches-bin/"
savepic = "savepic"
batchsize = 256

def read_data(dir,type):
    labelBytes = 1
    witdthBytes = 32
    heightBytes = 32
    depthBytes = 3
    imageBytes = witdthBytes * heightBytes * depthBytes
    recordBytes = imageBytes + labelBytes
    data_files = tf.gfile.Glob(dir  + type+'_batch*.bin')
    filename_queue = tf.train.string_input_producer(data_files,shuffle=True)
    reader = tf.FixedLengthRecordReader(record_bytes=recordBytes)  # 按固定长度读取二进制文件
    key, value = reader.read(filename_queue)

    bytes = tf.decode_raw(value,tf.uint8)  # 解码为uint8,0-255 8位3通道图像
    label = tf.reshape(tf.cast(
       tf.strided_slice(bytes, [0], [labelBytes],[1]), tf.int32),[1])    # 分割label并转化为int32
    image = tf.strided_slice(bytes, [labelBytes], [labelBytes + imageBytes])
    SRCImg = tf.reshape(image,[depthBytes, heightBytes, witdthBytes])
    img = tf.transpose(SRCImg, [1, 2, 0])  # 调整轴的顺序，深度在后
    img = tf.cast(img, tf.float32) * (1. / 255) - 0.5#数据归一化
    img_batch, label_batch = tf.train.shuffle_batch([img, label],
                                                    batch_size= batchsize,
                                                    num_threads=4,
                                                    capacity=1000+3*batchsize,
                                                    min_after_dequeue=1000
                                                    )

    label_batch = tf.one_hot(label_batch, 10)
    label_batch = tf.reshape(label_batch,[-1,10])
    return img_batch,label_batch

if __name__ == '__main__':
    img, label = read_data(dir,"data")
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        coord=tf.train.Coordinator()
        threads= tf.train.start_queue_runners(sess=sess,coord=coord)
        for i in range(5):
            imgs,labs = sess.run([img,label])#在会话中取出imgs和labs
            # imgs=Image.fromarray(imgs, 'RGB')#这里Image是之前提到的
            # plt.imshow(imgs)#显示图像
            # plt.show()
            # imgs.save(savepic+"/"+str(i)+'_''Label_'+str(labs)+'.jpg')#存下图片
            print(imgs.shape,labs.shape)
        coord.request_stop()
        coord.join(threads)