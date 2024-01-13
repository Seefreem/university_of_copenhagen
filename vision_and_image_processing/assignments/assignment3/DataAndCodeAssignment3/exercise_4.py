import numpy as np
import ps_utils
import numpy.linalg as la
import matplotlib.pyplot as plt
from ps_utils import *

# 4 （1）
# read shiny_vase.mat data
I, mask, S = ps_utils.read_data_file('shiny_vase.mat')

# get indices of non zero pixels in mask
nz = np.where(mask > 0)
m,n = mask.shape

# for each mask pixel, collect image data
J = np.zeros((3, len(nz[0])))
for i in range(3):
    Ii = I[:,:,i]
    J[i,:] = Ii[nz]

# solve for M = rho*N
iS = la.inv(S)
M = np.dot(iS, J)

# get albedo as norm of M and normalize M
Rho = la.norm(M, axis=0)
albedo_image = np.zeros((m, n))
albedo_image[nz] = Rho[:]

#albedo image
plt.imshow(albedo_image, cmap='gray')
plt.title('Albedo Image')
plt.show()

#normalize M 
N = M/np.tile(Rho, (3,1))
n1 = np.zeros((m,n))
n2 = np.zeros((m,n))
n3 = np.ones((m,n))
rho = np.ones((m,n))
n1[nz] = N[0,:]
n2[nz] = N[1,:]
n3[nz] = N[2,:]
rho[nz] = Rho[:]
# print(n1[0], n2[0], n3[0])

#normal image
_,(ax1,ax2,ax3) = plt.subplots(1,3)
ax1.imshow(n1)
ax2.imshow(n2)
ax3.imshow(n3)
plt.show()

#depth image
z = ps_utils.unbiased_integrate(n1, n2, n3, mask)
ps_utils.display_surface(z)


#4 （2）
# read shiny_vase.mat data
I, mask, S = read_data_file('shiny_vase.mat')

"""print(I.shape)
print(mask.shape)
print(S.shape)"""

# Get the size of the images and number of views
m, n, k = I.shape
albedo_matrix = np.zeros((mask.shape[0], mask.shape[1]))
normals_matrix = np.zeros((mask.shape[0], mask.shape[1], 3))

# traverse every pixel
for row in range(m):
    for col in range(n):
        if mask[row, col]:
            # construct data for RANSAC
            data = (I[row, col, :], S)
            modulated_normal, inliers, best_fit = ransac_3dvector(data,1.0,
                                                                  max_data_tries=100, max_iters=1000, p=0.9,
                                                                  det_threshold=1e-1, verbose=2)
            # calculate albedo
            
            albedo_matrix[row, col] = np.linalg.norm(modulated_normal)
            # calculate normal —— normalization
            normals_matrix[row, col, :] = modulated_normal / (albedo_matrix[row, col] + 0.0001)

plt.imshow(albedo_matrix,cmap='gray')
plt.title('image of albedo')
plt.show()

#plot n1, n2, n3 
_,(ax1,ax2,ax3) = plt.subplots(1,3)

ax1.imshow(normals_matrix[:,:,0])
ax1.set_title('n1')
ax2.imshow(normals_matrix[:,:,1])
ax2.set_title('n2')
ax3.imshow(normals_matrix[:,:,2])
ax3.set_title('n3')
plt.show()

#depth image (doesn't show in the document)
a = unbiased_integrate(normals_matrix[:,:,0], normals_matrix[:,:,1], normals_matrix[:,:,2], mask)

ps_utils.display_surface(a)


#use smooth_normal_field(iter=4) and plot the normal image and depth image
(n1, n2, n3) = smooth_normal_field(normals_matrix[:,:,0], normals_matrix[:,:,1], normals_matrix[:,:,2], mask, iters=4)
_,(ax1,ax2,ax3) = plt.subplots(1,3)

ax1.imshow(n1)
ax1.set_title('n1')
ax2.imshow(n2)
ax2.set_title('n2')
ax3.imshow(n3)
ax3.set_title('n3')
z = unbiased_integrate(n1, n2, n3, mask)
display_surface(z)

#use smooth_normal_field(iter=150) and plot the normal image and depth image
(n1, n2, n3) = smooth_normal_field(normals_matrix[:,:,0], normals_matrix[:,:,1], normals_matrix[:,:,2], mask, iters=150)
_,(ax1,ax2,ax3) = plt.subplots(1,3)
ax1.imshow(n1)
ax1.set_title('n1')
ax2.imshow(n2)
ax2.set_title('n2')
ax3.imshow(n3)
ax3.set_title('n3')
z = unbiased_integrate(n1, n2, n3, mask)
display_surface(z)
