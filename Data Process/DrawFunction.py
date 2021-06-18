# -*- coding: utf-8 -*-
"""
Created on Sun Mar 11 20:41:57 2018

@author: brucelau
"""

import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-10, 10, 100)
y_sigmoid = 1/(1+np.exp(-x))
y_tanh = (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))

plt.figure(dpi=200)
# # plot sigmoid
# ax = fig.add_subplot(221)
plt.plot(x, y_tanh)
plt.grid()
# # plt.set_title('(a) Sigmoid')

# # plot tanh
# ax = fig.add_subplot(222)
# ax.plot(x,y_tanh)
# ax.grid()
# ax.set_title('(b) Tanh')
#
# plot relu
# ax = fig.add_subplot(223)
# y_relu = np.array([0*item  if item<0 else item for item in x ])
# plt.plot(x, y_relu)
# plt.grid()
# ax.set_title('(c) ReLu')
#
# #plot leaky relu
# ax = fig.add_subplot(224)
# y_relu = np.array([0.2*item  if item<0 else item for item in x ])
# ax.plot(x,y_relu)
# ax.grid()
# ax.set_title('(d) Leaky ReLu')
#
# plt.tight_layout()

plt.savefig('tanh.png')

plt.show()

