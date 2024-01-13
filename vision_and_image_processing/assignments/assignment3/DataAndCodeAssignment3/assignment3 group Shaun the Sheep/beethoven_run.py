# -*- coding: utf-8 -*-
"""
Run Beethoven reconstruction code 

Author: Francois Lauze, University of Copenhagen
Date: Mon Jan  4 14:11:54 2016
"""


import numpy as np
import ps_utils
import numpy.linalg as la
import matplotlib.pyplot as plt

# read Beethoven data
I, mask, S = ps_utils.read_data_file('DataAndCodeAssignment3/Buddha')

# get indices of non zero pixels in mask
nz = np.where(mask > 0)
m,n = mask.shape

# for each mask pixel, collect image data
J = np.zeros((3, len(nz[0])))
for i in range(3):
    Ii = I[:,:,i]
    J[i,:] = Ii[nz] # (3, nz)


# solve for M = rho*N
iS = la.inv(S)  # (3, nz)
M = np.dot(iS, J)  # (3, nz)
print(iS.shape, M.shape)

# get albedo as norm of M and normalize M
Rho = la.norm(M, axis=0)
N = M/np.tile(Rho, (3,1))

n1 = np.zeros((m,n))
n2 = np.zeros((m,n))
n3 = np.ones((m,n))
rho = np.ones((m,n))
n1[nz] = N[0,:]
n2[nz] = N[1,:]
n3[nz] = N[2,:]
rho[nz] = Rho[:]


for i in n1.flatten():
    if abs(i) > 0.99:
        print(i)
_,(ax1,ax2,ax3) = plt.subplots(1,3)
ax1.imshow(n1)
ax2.imshow(n2)
ax3.imshow(n3)
plt.show()

z = ps_utils.unbiased_integrate(n1, n2, n3, mask)
for i in z.flatten():
    if i > 0 :
        print(i)
ps_utils.display_surface(z)


