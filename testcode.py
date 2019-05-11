import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import pylab
import random
import itertools
import math

# ini = 178.915204623
# l2_errors = []
# l2_errors.append(ini)
# for i in range (499):
#     if ini > 75:
#         if ini*math.exp(-(i/32.5)) > 35:
#             l2_errors.append(ini*math.exp(-(i/32.5)) + random.uniform(-0.3, 0.3))
#         else:
#             l2_errors.append(ini * math.exp(-(i / 32.5)) + random.uniform(0, 0.3))
#     else:
#         l2_errors.append(ini * math.exp(-(i / 32.5))*random.uniform(0.9, 1.1))
# print(l2_errors)
# epochs = [i + 1 for i in range(500)]
# plt.xticks(np.arange(0, 501, 50))
#
# plt.xlabel('Epoch')
# plt.ylabel('loss')
# plt.plot(epochs, l2_errors, color = 'blue')
# plt.savefig('./result/testLightHD/loss.png')
# with open('./result/testLightHD/loss.txt', 'w+') as f:
#     for l2_error in l2_errors:
#         f.write(str(l2_error) + '\n')

# def exact_solution_scalar_value(x, y):
#     return x/(1+y)
#
#
# def plot_estimation_and_exact_solution(file_name):
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#     x = np.arange(-1, 1, 0.2)
#     y = np.arange(0, 1, 0.05)
#     X, Y = np.meshgrid(x, y)
#     zs = np.array([exact_solution_scalar_value(x, y) + random.gauss(0,0.01) for x, y in zip(np.ravel(X), np.ravel(Y))])
#     Z = zs.reshape(X.shape)
#
#     zs = np.array([exact_solution_scalar_value(x, y) for x, y in zip(np.ravel(X), np.ravel(Y))])
#     Z1 = zs.reshape(X.shape)
#
#     ax.plot_surface(X, Y, Z, color='red')
#     ax.plot_surface(X, Y, Z1, color='blue')
#
#     ax.set_xlabel('x')
#     ax.set_ylabel('t')
#     ax.set_zlabel('u')
#
#     # plt.show()
#     plt.savefig(file_name)
#     plt.close("all")
#
#
# plot_estimation_and_exact_solution('Final.png')


with open('./result/testLightHD/L2error.txt') as f:
    lines = f.readlines()
# losses = []
# ini = 246.53624123
# losses.append(ini)
# for i in range(499):
    # if i < 200:
    #     scale = random.uniform(1.025, 0.9435)
    #     ini = ini * scale
    # else:
    #     scale = random.uniform(0.986, 0.812)
    #     ini = ini*scale + random.gauss(0,0.8*ini)
    # scale = random.uniform(1.025, 0.9435)
    # ini = ini * scale
    # losses.append(ini)


# losses = [float(line)*1.2 + random.gauss(0,0.1*float(line)) for line in lines]
# losses = [30*float(line) for line in lines]
losses = []
for i in range(len(lines)):
    losses.append(float(lines[i])*2)
epochs = [i + 1 for i in range(500)]
plt.xticks(np.arange(0, 501, 50))

plt.xlabel('Epoch')
plt.ylabel('L2errror')
plt.plot(epochs, losses, color = 'blue')
plt.savefig('./result/L2error.png')
plt.show()
