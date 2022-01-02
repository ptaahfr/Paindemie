import matplotlib.pyplot as plt
import numpy as np

data = np.genfromtxt('data.csv', delimiter=',')
plt.plot(data[:,0], data[:,1])
plt.show()