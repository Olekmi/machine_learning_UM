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

# sigmoid function
def sigmoid(x):
    
    return 1/(1+exp(-x))


def neural_network ():
    error_3 = 1
    # initializing weights for each node
    nodes_weights = random_weights(8,3)
    # output layer weights
    output_layer_weights = random_weights(4,8)
    for i in range(50):
        print("iteration:",i)
        print("mean error",abs(np.mean(error_3)))
        if abs(np.mean(error_3)) > 0.05:

            # hidden nodes' inputs 
            hidden_node_input_1 = (nodes_weights[:,0].T*input_data_1).sum()
            hidden_node_input_2 = (nodes_weights[:,1].T*input_data_1).sum()
            hidden_node_input_3 = (nodes_weights[:,2].T*input_data_1).sum()
            
            # outputs of particular nodes in  hidden layer
            output_hidden_node_1 = sigmoid(hidden_node_input_1)
            output_hidden_node_2 = sigmoid(hidden_node_input_2)
            output_hidden_node_3 = sigmoid(hidden_node_input_3)

            # hidden layer output( array)
            bias = 1
            output_hidden_nodes = np.array([bias, output_hidden_node_1,output_hidden_node_2,output_hidden_node_3])
            output_hidden_nodes = np.array(output_hidden_nodes).reshape(1,-1)

            # outputs
            output_1 = np.dot(output_hidden_nodes,output_layer_weights)

            # “errors” of nodes in layer 3
            error_3 = output_1 - input_data_1

            # “errors” of nodes in layer 2
            # for second layer for first node
            error_2_1 = output_hidden_nodes[0][0] * (1-output_hidden_nodes[0][0]) # it will be zero 
            error_2_2 = output_hidden_nodes[0][1]* (1-output_hidden_nodes[0][1])* (output_layer_weights[1,0]*error_3[0][0] + output_layer_weights[1,1]*error_3[0][1]  +
                                                                            output_layer_weights[1,2]*error_3[0][2]  + output_layer_weights[1,3]*error_3[0][3]  + output_layer_weights[1,4]*error_3[0][4]  
                                                                            + output_layer_weights[1,5]*error_3[0][5]  + output_layer_weights[1, 6]*error_3[0][6]  + output_layer_weights[1, 7]*error_3[0][7] )

            # w koncu cos sie udalo 


            error_2_3 = output_hidden_nodes[0][2]* (1-output_hidden_nodes[0][2])* (output_layer_weights[2,0]*error_3[0][0] + output_layer_weights[2,1]*error_3[0][1]  +
                                                                            output_layer_weights[2,2]*error_3[0][2]  + output_layer_weights[2,3]*error_3[0][3]  + output_layer_weights[2,4]*error_3[0][4]  
                                                                            + output_layer_weights[2,5]*error_3[0][5]  + output_layer_weights[2, 6]*error_3[0][6]  + output_layer_weights[2, 7]*error_3[0][7] ) 

            error_2_4 = output_hidden_nodes[0][3]* (1-output_hidden_nodes[0][3])* (output_layer_weights[3,0]*error_3[0][0] + output_layer_weights[3,1]*error_3[0][1]  +
                                                                            output_layer_weights[3,2]*error_3[0][2]  + output_layer_weights[3,3]*error_3[0][3]  + output_layer_weights[3,4]*error_3[0][4]  
                                                                            + output_layer_weights[3,5]*error_3[0][5]  + output_layer_weights[3, 6]*error_3[0][6]  + output_layer_weights[3, 7]*error_3[0][7] ) 
            error_2= [error_2_1, error_2_2, error_2_3, error_2_4]
            error_2 = np.array(error_2).reshape(1,-1)
            error_3 = error_3.reshape(-1,1)
            # update weights 

            output_layer_weights = error_3*output_hidden_nodes

            output_layer_weights = output_layer_weights.transpose()

            # setting alpha 

            alpha = 0.1
            output_layer_weights = output_layer_weights - alpha* output_layer_weights
            # te same wyliczenia:
            # nie bierzemy pod uwagé 1 nodu z hidden layer !!


            part_1_prim = error_2[0][1:4] * input_data_1.transpose() 
            # aktualizujemy wagi dla polaczen z input layer do hidden layer
            nodes_weights = nodes_weights - alpha*part_1_prim 
            # now we have all of the weights updated so we can pass the forward propagation again and predict new results - this time for
            # the second input argument

            # input_data_2 = np.array([0,1,0,0,0,0,0,0]).reshape(1,-1)

            # hidden_node_input_1 = (nodes_weights[:,0].T*input_data_2 ).sum()
            # hidden_node_input_2 = (nodes_weights_1[:,1].T *input_data_2).sum()
            # hidden_node_input_3 = (nodes_weights_1[:,2].T* input_data_2 ).sum()

            # # outputs of particular nodes in  hidden layer

            # output_hidden_node_1 = sigmoid(hidden_node_input_1)
            # output_hidden_node_2 = sigmoid(hidden_node_input_2)
            # output_hidden_node_3 = sigmoid(hidden_node_input_3)

            # # hidden layer output( array)
            # # vectorizing it 
            # bias = 1
            # output_hidden_nodes_2 = np.array([bias, output_hidden_node_1,output_hidden_node_2,output_hidden_node_3])
    return output_1, error_3

output_1, error_3 = neural_network()
print(output_1)
print(error_3)




