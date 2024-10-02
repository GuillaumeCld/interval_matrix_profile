import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_context(rc={"lines.linewidth": 4})


n_period = 2
period = 2 * np.pi 
n = 100

# Create the figure and specify a 2x2 grid using gridspec
fig = plt.figure(figsize=(20, 5))
# gs = fig.add_gridspec(2, 2, wspace=0.4, hspace=0.2)

# First plot: spans both columns in the first row
# ax1 = fig.add_subplot(gs[0, :])  # Span both columns with [:]

for i in range(n_period):
    x = np.linspace(i * period, (i + 0.5) * period, n)
    y = np.cos(x)
    # ax1.plot(x, y, color='blue', label='Season 1')
    plt.plot(x, y, color='blue', label='Season 1')

    x = np.linspace((i + 0.5) * period, (i + 1) * period, n)
    y = np.cos(x)
    # ax1.plot(x, y, color='orange', label='Season 2')
    plt.plot(x, y, color='orange', label='Season 2')

# ax1.axis('off')
# ax1.legend(['Season 1', 'Season 2'], ncol=2, loc='lower center', bbox_to_anchor=(0.5, -0.15), prop={'size': 18})

plt.axis('off')
# plt.legend(['Season 1', 'Season 2'], ncol=2, loc='lower center', bbox_to_anchor=(0.5, -0.15), prop={'size': 18})


# # Second plot: left column, second row
# ax2 = fig.add_subplot(gs[1, 0])

# for i in range(n_period):
#     x = np.linspace(i * period, (i + 0.5) * period, n)
#     y = np.cos(x)
#     ax2.plot(x - i * np.pi, y, color='blue')

# ax2.axis('off')

# # Third plot: right column, second row
# ax3 = fig.add_subplot(gs[1, 1])

# for i in range(n_period):
#     x = np.linspace((i + 0.5) * period, (i + 1) * period, n)
#     y = np.cos(x)
#     ax3.plot(x - (i + 1) * np.pi, y, color='orange')

# ax3.axis('off')

# Display the subplots
# plt.tight_layout()
plt.savefig('cos.png', bbox_inches='tight', pad_inches=0)
plt.show()