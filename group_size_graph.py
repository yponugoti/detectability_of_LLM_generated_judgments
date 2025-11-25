import matplotlib.pyplot as plt
import numpy as np

x = np.array([1, 2, 4, 8, 16])
helpsteer2 = np.array([90.6, 96.6, 99.4, 100, 100])
helpsteer3 = np.array([63.0, 67.0, 71.3, 73.3, 78.1])
neurips = np.array([84.8, 87.3, 91.4, 96.8, 99.7])
antique = np.array([68.4, 74.6, 82.6, 88.7, 94.6])

plt.figure(figsize=(8, 6))

plt.plot(x, helpsteer2, label='helpsteer2', marker='o', color = 'b')
plt.plot(x, helpsteer3, label='helpsteer3', marker='o', color = 'r')
plt.plot(x, neurips, label='NeurIPS', marker='o', color = 'g')
plt.plot(x, antique, label='ANTIQUE', marker='o', color = 'y')

plt.xlabel("k")
plt.ylabel("F1 (%)")
plt.title("Group Size")

plt.legend()
plt.grid(True)
plt.show()
