import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np


n = 50
width  = 3
height = 20



fig = plt.figure(figsize=(width, height))
ax = fig.add_subplot(111, aspect='equal')
plt.ylim(max(plt.ylim()), min(plt.ylim()))

metarows = n // height 
if n % height > 0:
    metarows += 1

nb_block = n // width + 1
if n % width > 0:
    nb_block += 1


missing = int(np.ceil((n  - (nb_block - int(np.ceil(height / width)) )*width)/width))

for row in range(metarows):
    block_shift = (row+1) * height
    diagonal_shift = int(np.ceil(block_shift / width))
    print(missing)
    for block in range(nb_block+missing):
        x1 = -diagonal_shift * width + row*height + block * width
        x2 = x1 + width
        x3 = x1 + height
        x4 = x1 + height + width

        y1 = row*height
        y2 = y1
        y3 = (row+1)*height
        y4 = y3

        x = [x1, x2, x4, x3]
        y = [y1, y2, y4, y3]

        ax.add_patch(patches.Polygon(xy=list(zip(x,y)), fill=False))

        # Left Triangle
        if x4 <= width:
            xt1= x4
            xt2 = 0 
            xt3 = 0

            yt1 = y4
            yt2 = y4
            yt3 = y4-x4

            xt = [xt1, xt2, xt3]
            yt = [yt1, yt2, yt3]
            ax.add_patch(patches.Polygon(xy=list(zip(xt,yt)), fill=True))
        # Right Triangle
        if x1 < n and x1 >= n-width and x3>=n:
            xt1 = x1
            xt2 = n 
            xt3 = n

            yt1 = y1
            yt2 = y1
            yt3 = y1+(n-x1)

            xt = [xt1, xt2, xt3]
            yt = [yt1, yt2, yt3]
            ax.add_patch(patches.Polygon(xy=list(zip(xt,yt)), fill=True))
        # Quadrangle
        if x1 < 0 and x4 > height and x4 < width:
            xp1 = 0
            xp2 = x2
            xp3 = x4
            xp4 = 0

            yp1 = y1
            yp2 = y2
            yp3 = y4
            yp4 = y3

            xp = [xp1, xp2, xp3, xp4]
            yp = [yp1, yp2, yp3, yp4]
            ax.add_patch(patches.Polygon(xy=list(zip(xp,yp)), fill=True, color='green'))
        # Polygon
        if x1 < 0 and x4 > height and x4 >= width:
            xp1 = 0
            xp2 = x2
            xp3 = x4
            xp4 = x3
            xp5 = 0

            yp1 = y1
            yp2 = y2
            yp3 = y4
            yp4 = y3
            yp5 = y4-(x4 - width)

            xp = [xp1, xp2, xp3, xp4, xp5]
            yp = [yp1, yp2, yp3, yp4, yp5]
            ax.add_patch(patches.Polygon(xy=list(zip(xp,yp)), fill=True, color='orange'))

plt.plot([0, n, n, 0, 0], [0, 0, n, n, 0], 'k-', color='red')
plt.axis('equal')
plt.show()