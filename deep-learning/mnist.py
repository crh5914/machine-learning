from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
#load mnist dataset
mnist = input_data.read_data_sets('MNIST_data/',one_hot=True)
#explore the dataset
#print(mnist.train.images.shape,mnist.train.labels.shape) #(55000,784),(55000,10)
#print(mnist.test.images.shape,mnist.test.labels.shape)#(10000,784),(10000,10)
#print(mnist.validation.images.shape,mnist.validation.labels.shape)#(5000,784),(5000,10)
#define input
x = tf.placeholder(tf.float32,[None,784])
y = tf.placeholder(tf.float32,[None,10])
#define parameters
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))
#define output and loss function
y_ = tf.nn.softmax(tf.matmul(x,W) + b)
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y*tf.log(y_),axis=[1]))#reduce_mean calculate the mean of batchs
trainer = tf.train.GradientDescentOptimizer(0.3).minimize(cross_entropy)#0.01->88.72%, 0.1->90%,0.3->92.27%
initializer = tf.global_variables_initializer()
#open session
with tf.Session() as sess:
	sess.run(initializer)
	epoch = 10 #trainning times on the whole dataset
	batch_size = 100 #fetch 100 sample to train every time
	for i in range(epoch):
		for k in range(550):
			batch_x,batch_y = mnist.train.next_batch(batch_size)
			sess.run(trainer,feed_dict={x:batch_x,y:batch_y})
	#test
	correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
	accuray = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
	print(sess.run(accuray,feed_dict={x:mnist.test.images,y:mnist.test.labels}))