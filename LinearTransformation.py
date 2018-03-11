import tensorflow as tf
import numpy as np
import pickle
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-s","--sourcepath",required=True)
args = vars(ap.parse_args())

pickle_in_1 = open(args["sourcepath"])
data = pickle.load(pickle_in_1)

total_samples = len(data)
total_embedding = len(data[0])
input_dimension = len(data[0][0])
output_dimension = 128

training_inputs = tf.placeholder(shape=[1,1024],dtype=tf.float32)
weights = tf.Variable(tf.random_normal([input_dimension,output_dimension],stddev=1.0))
output_op = tf.matmul(training_inputs,weights)

init_op = tf.global_variables_initializer()
sess = tf.Session()

output = []
for i in range(total_samples):
 inner_output = []
 for j in range(total_embedding):
   input_data = np.reshape(data[i][j],(1,input_dimension))
   sess.run(init_op)
   output_data = np.reshape(sess.run(output_op,feed_dict = {training_inputs : input_data}),(output_dimension,1))
   inner_output.append(output_data)
 output.append(inner_output)

output_file = args["sourcepath"]
pickle_out_1 = open(output_file[0:output_file.find('.')] + "-Reduced.pickle","w+")
pickle.dump(output,pickle_out_1)
