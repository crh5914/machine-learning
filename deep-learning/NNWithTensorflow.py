import tensorflow as tf
import numpy as np
#define neural networklayer
def add_layer(input,insize,outsize,active_func=None):
	weights = tf.Variable(tf.random_normal([insize,outsize]))
	bias = tf.Variable(tf.zeros([1,outsize])+0.1)
	mat_plus = tf.matmul(input,weights) + bias
	if active_func is None:
		output = mat_plus
	else:
		output = active_func(mat_plus)
	return output,weights,bias

#input data
x_data = np.linspace(-1,1,300)[:,None]
#radom noise
noise = np.random.normal(0,0.05,x_data.shape)
#expected output data
y_data = np.square(x_data)- 0.5 + noise

#define tensors
xs = tf.placeholder(tf.float32,[None,1])
ys = tf.placeholder(tf.float32,[None,1])

l1,weights1,bias1 = add_layer(xs,1,10,active_func=tf.nn.relu)
prediction,weights2,bias2= add_layer(l1,10,1,active_func=None)
#define the loss
loss = tf.reduce_mean(tf.square(prediction-ys))
#define optimizer
optimizer = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

init = tf.global_variables_initializer()
#open session
with tf.Session() as sess:
	#init variable
	sess.run(init)
	for i in range(1000):
		sess.run(optimizer,feed_dict={xs:x_data,ys:y_data})
		if i%50 == 0:
			print(sess.run([loss,weights1,weights2],feed_dict={xs:x_data,ys:y_data}))
	


 