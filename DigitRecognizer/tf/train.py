import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np
import os

# train_x=pd.read_csv('data/train.csv')
# train_y=train_x.pop('label')
# x_train, x_test, y_train, y_test = train_test_split(train_x, train_y, test_size=0.33, random_state=42)

def dense_to_one_hot(labels_dense, num_classes):
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot

def one_hot_to_data(data):
    index_=np.where(data.ravel()==1)
    index=index_[0]%10
    return index.T

#初始化权重函数
def weight_variable(shape):
    initial = tf.truncated_normal(shape,stddev=0.1);
    return tf.Variable(initial)

#初始化偏置项
def bias_variable(shape):
    initial = tf.constant(0.1,shape=shape)
    return tf.Variable(initial)

#定义卷积函数
def conv2d(x,w):
    return tf.nn.conv2d(x,w,strides=[1,1,1,1],padding='SAME')

#定义一个2*2的最大池化层
def max_pool_2_2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

def load_data(filename, train_data=True, split=0.9):
    data_frame = pd.read_csv(filename)
    # (42000, 785)
    print(data_frame.shape)

    train_data_len = data_frame.shape[0]
    train_data_split = int(train_data_len*split)
    print(train_data_split)

    train_x = data_frame.iloc[:train_data_split, 1:].values
    train_x = train_x.astype(np.float)
    train_x = np.multiply(train_x, 1.0/255.0)

    train_y = data_frame.iloc[:train_data_split, 0].values
    train_y = dense_to_one_hot(train_y,10)

    validate_x = data_frame.iloc[train_data_split:, 1:].values
    validate_x = validate_x.astype(np.float)
    validate_x = np.multiply(validate_x, 1.0/255.0)

    validate_y = data_frame.iloc[train_data_split:, 0].values
    validate_y = dense_to_one_hot(validate_y,10)

    print(train_x.shape)
    print(train_y.shape)
    print(validate_x.shape)
    print(validate_y.shape)
    print(validate_y)
    return  train_x, train_y, validate_x, validate_y

def load_predict(filename):
    data_frame = pd.read_csv(filename)
    print(data_frame.shape)
    predict_data_len = data_frame.shape[0]
    predict_x = data_frame.iloc[:predict_data_len, :].values
    predict_x = predict_x.astype(np.float)
    predict_x = np.multiply(predict_x, 1.0/255.0)
    print(predict_x.shape)
    return predict_x

x_train, y_train, x_test, y_test  = load_data('data/train.csv',True,1)
x_predict=load_predict('data/test.csv')

if __name__ == "__main__":
    #定义输入变量
    x = tf.placeholder("float",shape=[None,784])
    #定义输出变量
    y_ = tf.placeholder("float",shape=[None,10])
    #初始化权重,第一层卷积，32的意思代表的是输出32个通道
    # 其实，也就是设置32个卷积，每一个卷积都会对图像进行卷积操作
    w_conv1 = weight_variable([5,5,1,32])
    #初始化偏置项
    b_conv1 = bias_variable([32])
    #将输入的x转成一个4D向量，第2、3维对应图片的宽高，最后一维代表图片的颜色通道数
    # 输入的图像为灰度图，所以通道数为1，如果是RGB图，通道数为3
    # tf.reshape(x,[-1,28,28,1])的意思是将x自动转换成28*28*1的数组
    # -1的意思是代表不知道x的shape，它会按照后面的设置进行转换
    x_image = tf.reshape(x,[-1,28,28,1])
    # 卷积并激活
    h_conv1 = tf.nn.relu(conv2d(x_image,w_conv1) + b_conv1)
    #池化
    h_pool1 = max_pool_2_2(h_conv1)
    #第二层卷积
    #初始权重
    w_conv2 = weight_variable([5,5,32,64])
    #初始化偏置项
    b_conv2 = bias_variable([64])
    #将第一层卷积池化后的结果作为第二层卷积的输入
    h_conv2 = tf.nn.relu(conv2d(h_pool1,w_conv2) + b_conv2)
    #池化
    h_pool2 = max_pool_2_2(h_conv2)
    # 设置全连接层的权重
    w_fc1 = weight_variable([7*7*64,1024])
    # 设置全连接层的偏置
    b_fc1 = bias_variable([1024])
    # 将第二层卷积池化后的结果，转成一个7*7*64的数组
    h_pool2_flat = tf.reshape(h_pool2,[-1,7*7*64])
    # 通过全连接之后并激活
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat,w_fc1) + b_fc1)
    # 防止过拟合
    keep_prob = tf.placeholder("float")
    h_fc1_drop = tf.nn.dropout(h_fc1,keep_prob)

    #输出层
    w_fc2 = weight_variable([1024,10])
    b_fc2 = bias_variable([10])

    y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop,w_fc2) + b_fc2)

    #日志输出，每迭代100次输出一次日志
    #定义交叉熵为损失函数
    cross_entropy = -tf.reduce_sum(y_ * tf.log(y_conv))
    #最小化交叉熵
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    #计算准确率
    correct_prediction = tf.equal(tf.argmax(y_conv,1),tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,"float"))
    saver = tf.train.Saver()
    sess = tf.Session()
    sess.run(tf.initialize_all_variables())
    # 下载minist的手写数字的数据集
    # mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    # for i in range(20000):
    #     batch = mnist.train.next_batch(50)
    #     if i % 100 == 0:
    #         train_accuracy = accuracy.eval(session=sess,feed_dict={x:batch[0],y_:batch[1],keep_prob:1.0})
    #         print("step %d,training accuracy %g"%(i,train_accuracy))
    # print(x_train)
    # train_step.run(session = sess,feed_dict={x:[x_train],y_:[y_train],keep_prob:0.5})

    batch = 50
    train_size = x_train.shape[0]
    range_num=int(train_size/batch)+1
    print(range_num)
    for i in range(10000):
        start = i*batch % train_size
        end = (i+1)*batch % train_size
        # print(start, end)
        if start > end:
            start=0
        # if end >train_size:
        #     end=train_size

        batch_x = x_train[start:end]
        batch_y = y_train[start:end]  
        # print(batch_x)
        if i % 100 == 0:
            train_accuracy = accuracy.eval(session=sess,feed_dict={x:batch_x,y_:batch_y,keep_prob:1.0})
            print("step %d,training accuracy %g"%(i,train_accuracy))
        train_step.run(session = sess,feed_dict={x:batch_x,y_:batch_y,keep_prob:0.5})

    model_dir = "mnist"
    model_name = "digi"
    # 保存模型
    saver.save(sess, os.path.join(model_dir, model_name))

    x_p=x_predict[0:10]
    myPrediction = sess.run(y_conv,feed_dict={x:x_predict, keep_prob:1.0})
    label_test = np.argmax(myPrediction,axis=1)
    print(label_test)
    pd.DataFrame(label_test).to_csv('simpleNN2.csv')

    # print("test accuracy %g" % accuracy.eval(session=sess,feed_dict={
    #     x: x_test, y_: y_test, keep_prob: 1.0}))
    #test accuracy 0.9919

