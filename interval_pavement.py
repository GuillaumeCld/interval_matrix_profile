import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np


n = 100
width  = 10
height = 60

fig = plt.figure(figsize=(width, height))
ax = fig.add_subplot(111, aspect='equal')
plt.ylim(max(plt.ylim()), min(plt.ylim()))

metarows = n // height 
if n % height > 0:
    metarows += 1

nb_block = n // width
if n % width > 0:
    nb_block += 1

def_height = height

missing = int(np.ceil((n  - (nb_block - int(np.ceil(height / width)) )*width)/width))
for row in range(metarows):
    block_shift = (row+1) * height
    diagonal_shift = int(np.ceil(block_shift / width))
    for block in range(nb_block+missing):
        x1 = -diagonal_shift * width + row*height + block * width
        x2 = x1 + width
        x3 = x1 + height
        x4 = x1 + height + width

        y1 = row*def_height
        y2 = y1
        y3 = y1+height
        y4 = y3

        x = [x1, x2, x4, x3]
        y = [y1, y2, y4, y3]

        ax.add_patch(patches.Polygon(xy=list(zip(x,y)), fill=False))

plt.plot([0, n, n, 0, 0], [0, 0, n, n, 0], 'k-', color='red')
plt.axis('equal')
plt.show()