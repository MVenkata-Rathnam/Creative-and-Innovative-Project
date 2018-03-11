import pickle
import numpy as np
import argparse

ap = argparse.ArgumentParser()
ap.add_argument('-f',"--filepath",required=True)
args = vars(ap.parse_args())

pickle_in_1 = open(args["filepath"],"rb")
data = pickle.load(pickle_in_1)
print (len(data))
print (len(data[0]))
print (len(data[0][0]))
print (len(data[0][0][0]))
#print (data)

#pickle_in_2 = open(args["filepath"])
#class_info = pickle.load(pickle_in_2)
#print (len(class_info))
#print (class_info)

#pickle_in_3 = open(args["filepath"])
#file_names = pickle.load(pickle_in_3)
#print (file_names)
