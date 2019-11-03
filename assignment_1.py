import numpy as np 
from math import exp

def normal_distribution(rows,cols):
    mu, sigma = 0, 0.01 # mean and standard deviation
    s = np.random.normal(mu, sigma, cols)
    s_0 = s
    for i in range(rows-1):
        s = np.vstack([s, s_0])
    arrays = np.array(s)
    print(arrays)
    return arrays

def random_weights(rows,columns):#not used anymore
    epsilon = 10000
    # return np.random.rand(rows, columns)/epsilon
    return np.random.rand(rows, columns)/epsilon

# sigmoid function
def sigmoid(x):
    return 1/(1+exp(-x))

def neural_network ():
    # input_data_1 = np.array([1,0,0,0,0,0,0,0]).reshape(1,-1)
    input_data_1_in = np.eye(8,dtype=int)
    error_3 = 1
    alpha_0 = 0.04
    decay_rate = 0.5
    iterations = 100
    stop_criterion = 0.1
    # initializing weights for each node
    nodes_weights = normal_distribution(8,3)
    # output layer weights
    output_layer_weights = normal_distribution(4,8)
    for i in range(iterations):
        print("iteration:",i)
        if np.mean(np.absolute(error_3)) < stop_criterion:
                break
        for j in range(8):
            alpha = (1/(1+decay_rate*i))*alpha_0
            input_data_1 = input_data_1_in[j].reshape(1,-1)
            
            print("mean error",np.mean(np.absolute(error_3)))
            if np.mean(np.absolute(error_3)) < stop_criterion:
                break

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
            error_3 = (output_1 - input_data_1)#+par_lambda*0.5*(output_layer_weights*output_layer_weights)

            # “errors” of nodes in layer 2
            # for second layer for first node
            error_2_1 = output_hidden_nodes[0][0] * (1-output_hidden_nodes[0][0]) # it will be zero 
            error_2_2 = output_hidden_nodes[0][1]* (1-output_hidden_nodes[0][1])* (output_layer_weights[1,0]*error_3[0][0] + output_layer_weights[1,1]*error_3[0][1]  +
                                                                            output_layer_weights[1,2]*error_3[0][2]  + output_layer_weights[1,3]*error_3[0][3]  + output_layer_weights[1,4]*error_3[0][4]  
                                                                            + output_layer_weights[1,5]*error_3[0][5]  + output_layer_weights[1, 6]*error_3[0][6]  + output_layer_weights[1, 7]*error_3[0][7] )

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
            output_layer_weights = output_layer_weights - alpha* output_layer_weights
            # te same wyliczenia:
            # nie bierzemy pod uwagé 1 nodu z hidden layer !!
            nodes_weights = error_2[0][1:4] * input_data_1.transpose() 
            # aktualizujemy wagi dla polaczen z input layer do hidden layer
            nodes_weights = nodes_weights - alpha*nodes_weights 
            
    return output_1, error_3

output_1, error_3 = neural_network()
print(output_1)
print("error", error_3)




