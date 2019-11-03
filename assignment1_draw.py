import assignment1_test as test
import numpy as np
import matplotlib.pyplot as plt

iterations = 7000
stop_criterion = 0.01
i_r, mean_r = test.neural_network(0.15,0.0001,iterations,stop_criterion)
i_b, mean_b = test.neural_network(0.1,0.0001,iterations,stop_criterion)
i_g, mean_g = test.neural_network(0.05,0.0001,iterations,stop_criterion)

plt.plot(i_r, mean_r, 'r-', label='alpha = 0.15')
plt.plot(i_b, mean_b, 'b-', label='alpha = 0.1')
plt.plot(i_g, mean_g, 'g-', label='alpha = 0.05')
plt.xlabel('iterations')
plt.ylabel('mean error')
plt.legend()
plt.show()