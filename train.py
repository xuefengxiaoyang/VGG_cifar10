import tensorflow as tf
import dataLoad
import VGG
import numpy as np


dir = "cifar-10-batches-bin/"
epoch = 50
batch_num = 50000


with tf.name_scope('input'):
    x = tf.placeholder(tf.float32,[None,32,32,3],name="x-input")
    y = tf.placeholder(tf.float32,[None,10],name="y-input")

prediction = VGG.VGG16(x)

with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=prediction))
    tf.summary.scalar('loss', loss)

with tf.name_scope('train_step'):
    train_step = tf.train.GradientDescentOptimizer(0.001).minimize(loss)

with tf.name_scope('accuracy'):
    correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(prediction,1))#argmax返回一维张量中最大的值所在的位置
    #计算准确率
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    tf.summary.scalar('accuracy', accuracy)

train_image,train_label = dataLoad.read_data(dir,"data")
test_image,test_label = dataLoad.read_data(dir,"test")

init = tf.global_variables_initializer()
merged = tf.summary.merge_all()

if __name__ == '__main__':
    with tf.Session() as sess:
        sess.run(init)
        writer = tf.summary.FileWriter('logs/',sess.graph)
        # 启动多线程
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        for i in range(epoch):
            for batch in range(batch_num):
                print(batch)

                train_batch_image,train_batch_label,test_batch_image,test_batch_label = sess.run([train_image,train_label,test_image,test_label])
                acc = sess.run(accuracy, feed_dict={x: test_batch_image, y: test_batch_label})
                print("Iter" + str(i) + ", Testing Accuracy " + str(acc))
                summary,_ =  sess.run([merged,train_step],feed_dict={x:train_batch_image,y:train_batch_label})
            writer.add_summary(summary, i)
            acc = sess.run(accuracy,feed_dict={x:test_batch_image,y:test_batch_label})
            print("Iter"+str(i)+", Testing Accuracy "+str(acc))
        saver.save(sess, 'net/my_net.ckpt')
        coord.request_stop()
        coord.join(threads)

