from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
mnist = input_data.read_data_sets('MNIST_data/',one_hot=True)
n_input = 784 #the number of input feactures
n_hidden = 300 #the number of hidden node
n_output = 10
x = tf.placeholder(tf.float32,[None,n_input])
y = tf.placeholder(tf.float32,[None,n_output])

W1 = tf.Variable(tf.truncated_normal([n_input,n_hidden],stddev=0.1))
b1 = tf.Variable(tf.zeros([n_hidden]))

W2 = tf.Variable(tf.zeros([n_hidden,n_output]))
b2 = tf.Variable(tf.zeros([n_output]))

keep_prob = tf.placeholder(tf.float32)
hidden1 = tf.nn.relu(tf.matmul(x,W1) + b1)
hidden1_dropout = tf.nn.dropout(hidden1,keep_prob)
y_ = tf.nn.softmax(tf.matmul(hidden1_dropout,W2)+ b2)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y*tf.log(y_),reduction_indices=[1])) #reuction_indices=[0]->raw add while [1]->column add
trainer = tf.train.AdagradOptimizer(0.3).minimize(cross_entropy)#0.1->97.53% 0.2->97.99% 0.3->97.96%
init = tf.global_variables_initializer()
with tf.Session() as sess:
	sess.run(init)
	for i in range(5000):  #5000,alpha=0.2,0.3->98%
		batch_x,batch_y = mnist.train.next_batch(100)
		sess.run(trainer,feed_dict={x:batch_x,y:batch_y,keep_prob:0.75})
	prediction = tf.equal(tf.argmax(y_,1),tf.argmax(y,1))
	accuray = tf.reduce_mean(tf.cast(prediction,tf.float32))
	print(sess.run(accuray,feed_dict={x:mnist.test.images,y:mnist.test.labels,keep_prob:1.0}))