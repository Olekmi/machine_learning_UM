# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 17:20:13 2019

@author: Olek
"""

import numpy as np 
from math import exp
# first experiment 

input_data_1 = np.array([1,0,0,0,0,0,0,0]).reshape(1,-1)

# creating weights 
 
def random_weights(rows,columns):
    return np.random.rand(rows, columns)

# initializing weights for each node

nodes_weights = random_weights(8,3)

# hidden nodes' inputs 

hidden_node_input_1 = (input_data_1 *nodes_weights[:,0]).sum()
hidden_node_input_2 = (input_data_1 *nodes_weights[:,1]).sum()
hidden_node_input_3 = (input_data_1 *nodes_weights[:,2]).sum()

# sigmoid function
def sigmoid(x):
    return 1/(1+exp(-x))
    
# outputs of particular nodes in  hidden layer

output_hidden_node_1 = sigmoid(hidden_node_input_1)
output_hidden_node_2 = sigmoid(hidden_node_input_2)
output_hidden_node_3 = sigmoid(hidden_node_input_3)

# hidden layer output( array)
bias = 1
output_hidden_nodes = np.array([bias, output_hidden_node_1,output_hidden_node_2,output_hidden_node_3]).reshape(1,-1)

# output layer weights

output_layer_weights = random_weights(4,8)

# outputs

output = np.dot(output_hidden_nodes,output_layer_weights)

