import assignment1_test as test
import numpy as np
import matplotlib.pyplot as plt

# iterations = 50000
# stop_criterion = 0.000
# decay_rate = 0.001

# i_r, mean_r = test.neural_network(0.1,decay_rate,iterations,stop_criterion)
# i_b, mean_b = test.neural_network(0.5,decay_rate,iterations,stop_criterion)
# i_g, mean_g = test.neural_network(1,decay_rate,iterations,stop_criterion)
# i_y, mean_y = test.neural_network(3,decay_rate,iterations,stop_criterion)

# plt.plot(i_r, mean_r, 'r-', label='alpha = 0.1')
# plt.plot(i_b, mean_b, 'b-', label='alpha = 0.5')
# plt.plot(i_g, mean_g, 'g-', label='alpha = 1')
# plt.plot(i_y, mean_y, 'y-', label='alpha = 3')
# plt.xlabel('iterations')
# plt.ylabel('mean error')
# plt.legend()
# plt.show()

alpha = []
error = []
min_error = []
for a in range(0,999,5):
    alpha_new = 0.01 + a/100 
    decay_rate = 0.001
    iterations = 6000
    stop_criterion = 0.00
    i_r, mean_r = test.neural_network(alpha_new,decay_rate,iterations,stop_criterion)
    alpha.append(alpha_new)
    min_error.append(min(mean_r))
    
plt.plot(alpha, min_error, 'r-', label='decay rate = 0.001')
plt.xlabel('alpha')
plt.ylabel('minimal error')
plt.legend()
plt.show()