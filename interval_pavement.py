import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd


df = pd.read_csv("Data/era5_daily_series_sst_60S-60N_ocean.csv", sep=",", skiprows=18)
df = df[df["status"] == "FINAL"]
df = df[(df['date'] >= '1979-01-01') & (df['date'] <= '2023-12-31')]
ts_date = df["date"].values.astype(np.datetime64)

list_ind = []

for year in range(1979, 2024):
    list_ind.append(np.where(ts_date == np.datetime64(str(year) + "-01-01"))[0][0])

with open("Data/periods_start_sst.txt", "w") as f:
    for item in list_ind:
        f.write(f"{item}\n")

I = 90
window_size = 14
n = ts_date.shape[0]
n_sequence = n - window_size + 1



if window_size > I//2:
    nb_block = len(list_ind)
else:
    nb_block = len(list_ind) + 1

fig = plt.figure()
ax = fig.add_subplot(111, aspect='equal')
plt.ylim(max(plt.ylim()), min(plt.ylim()))

for col in list_ind:
    plt.vlines(col, 0, n_sequence, color='purple', linestyles='dashed', alpha=0.3)


metarows = len(list_ind)


period_witdh = 0

width = I
for row in range(metarows):
    if row == metarows - 1:
        height = n_sequence - list_ind[row]
    else:
        height = list_ind[row+1] - list_ind[row]

    for col in range(nb_block):

        if col < len(list_ind):
            if col == len(list_ind) - 1:
                width = np.minimum(I, n_sequence - (list_ind[col]- I//2))
                period_witdh = n+1 - list_ind[col]
            else:
                width = I
                period_witdh = list_ind[col+1] - list_ind[col]
            x1 = list_ind[col] - I//2
        else:
            x1 = n+1 - I//2  
            period_witdh = -1
            if x1 >= n_sequence:
                break

        x2 = x1 + width
        x3 = x2 + height
        x4 = x1 + height

        y1 = list_ind[row]
        y2 = y1
        y3 = y1+height
        y4 = y3
        x = [x1, x2, x3, x4]
        y = [y1, y2, y3, y4]

        ax.add_patch(patches.Polygon(xy=list(zip(x,y)), fill=True, color='blue', alpha=0.5))

        if period_witdh > 0:
            if height > period_witdh:
                plt.hlines(y3, x4-1, x4, color='orange', alpha=1, lw=20)
                plt.hlines(y3, x3-1, x3, color='red', alpha=1, lw=20)
            elif height < period_witdh:
                plt.hlines(y3, x4, x4+1, color='red', alpha=1, lw=20)
                plt.hlines(y3, x3, x3+1, color='green', alpha=1, lw=20)


plt.plot([0, n, n, 0, 0], [0, 0, n, n, 0], 'k-', color='yellow')
plt.plot([0, n_sequence, n_sequence, 0, 0], [0, 0, n_sequence, n_sequence, 0], 'k-', color='red')
plt.axis('equal')
plt.show()