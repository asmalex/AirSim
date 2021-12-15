'''
This script is an example of how to extract a ground truth bounding box from a segmentation image. Since we made the obstacle black in our segmentation palatte, 
we can search for black pixels directly. Our bounding box is in the form of a mask for an RGB image, but we also extract the bounding coordinates of the box as well as its center.
'''


import numpy as np
from PIL import Image

# Make sure you use a segmentation image (which end with 0 instead of 1) when choosing a file
image = np.load("D:\\Redei\\AirSim\PythonClient\\computer_vision\\Dataset\\airsim_cv_mode\\2021_12_13_14_42_46\\0_0.npy", allow_pickle=True)

im = Image.fromarray(image)
im.show()

print(image.shape)

u = -1 # Highest point of the obstacle
b = -1 # Lowest point of the obstacle

for i in range(len(image)):
    for j in range(len(image[i])):
        if np.array_equal(image[i][j], [0,0,0]):
            if u == -1:
                u = i
            b = i

l = -1 # Leftmost point of the obstacle
r = -1 # Rightmost point of the obstacle

for j in range(len(image[0])):
    for i in range(len(image)):
        if np.array_equal(image[i][j], [0,0,0]):
            if l == -1:
                l = j
            r = j

print("upper bound, lower bound, leftmost bound, and rightmost bound: ", u,b,l,r)
i = u + (b-u)//2
j = l + (r-l)//2
print("box center: ", [i,j])

box_mask = np.zeros(image.shape)

for i in range(len(image)):
    for j in range(len(image[i])):
        if (i == u or i == b) and j >= l and j <= r:
            box_mask[i][j] = [1,1,1]
        elif (j == l or j == r) and i >= u and i <= b:
            box_mask[i][j] = [1,1,1]

im = Image.fromarray((255*box_mask).astype(np.uint8))
im.show()