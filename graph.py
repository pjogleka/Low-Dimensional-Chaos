import numpy as np
import matplotlib.pyplot as plt

x = np.array([-0.0795, -1.6330, -2.3094, -2.8284, -3.2660, -3.6515, -4, 4.3205, 4.6188, 4.8990, 5.1640, 5.4160, 5.6569, 5.8878, 6.1101, 6.3246, 6.5320, 6.7330, 6.9282, -7.1180, -7.3030, -7.4206])
y = x
y[-1] = -7.4027
z = np.zeros(22)
z_range = np.arange(1, 21)
z[1:21] = z_range
z[0] = 0.0024
z[-1] = 20.96
r = np.arange(1, 23)

plt.plot(r, np.abs(x), linestyle='-', linewidth=1.0, label = "x = y")
#plt.plot(r, y, linestyle='-', linewidth=1.0, label = "y")
plt.plot(r, z, linestyle='-', linewidth=1.0, label = "z")
plt.xlabel('r')
plt.ylabel('magnitude of fixed point')
plt.title('fixed points vs. r')
plt.legend()
plt.show()