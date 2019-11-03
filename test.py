import numpy as np 
from math import exp

def normal_distribution(rows,cols):
    mu, sigma = 0, 0.01 # mean and standard deviation
    s = np.random.normal(mu, sigma, cols)
    s_0 = s
    for i in range(rows-1):
        # s = np.vstack([s,s0])
        s = np.vstack([s,np.random.normal(mu, sigma, cols)])
    arrays = np.array(s)
    # print(arrays)
    return arrays 

# sigmoid function
def sigmoid(x):
    return 1/(1+np.exp(-x))

def neural_network ():
    input_data = np.eye(8,dtype=int)
    # initializing weights for each node
    nodes_weights = normal_distribution(8,3)
    # output layer weights
    output_layer_weights = normal_distribution(4,8)
   
    alpha_0 = 0.15
    delta_3 = 1
    decay_rate = 0.00001
    iterations = 10000
    stop_criterion = 0.01

    for i in range(iterations):
        print("iteration:",i)
        print("mean error",np.mean(np.absolute(delta_3)))
        if np.mean(np.absolute(delta_3)) < stop_criterion:
                break
        alpha = (1/(1+decay_rate*i))*alpha_0    
        hidden_node_input = np.dot(input_data ,nodes_weights)
        activation_hidden_layer = sigmoid(hidden_node_input)
        activation_hidden_layer  = activation_hidden_layer.T
        # hidden layer output( array)
        bias = np.array([1,1,1,1,1,1,1,1])
        output_hidden_nodes = np.vstack([bias, activation_hidden_layer])
        output_1 = sigmoid(np.dot(output_hidden_nodes.T,output_layer_weights))
        delta_3 = (output_1 - input_data)
        delta_2 = np.dot(output_layer_weights, delta_3) * output_hidden_nodes * (1 - output_hidden_nodes)
        # error_2= [error_2_1, error_2_2, error_2_3, error_2_4]
        # error_2 = np.array(error_2).reshape(1,-1)
        # error_3 = error_3.reshape(-1,1)
        # update weights 
        # output_layer_weights = output_layer_weights.transpose()
        output_layer_weights = output_layer_weights - alpha*np.dot( output_hidden_nodes, delta_3)
        # aktualizujemy wagi dla polaczen z input layer do hidden layer
        nodes_weights = nodes_weights.T - alpha*np.dot(delta_2[1:4,:], input_data.transpose() )
        nodes_weights = nodes_weights.T
    print(np.round(abs(output_1),3))
    print("error", delta_3)
    print("weights_2", np.round(nodes_weights,2))
    print("weights_3", np.round(output_layer_weights,2))
    return output_1, delta_3

    

neural_network ()

