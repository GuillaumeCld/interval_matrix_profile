import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import xarray as xr
sns.set_context("talk", font_scale=1.25)

def load_data(filename):
    data = []
    with open(filename, 'r') as f:
        for line in f:
            data.append(float(line.strip()))
    return np.array(data)

ds = xr.open_dataset("../Bordeaux_t2m_hourly_avg_1950_2023.nc")
date = ds.time.values.astype('datetime64[h]')
data = ds.t2m.values.astype(np.float64)
# Create a list with a time series per year
time_series_per_year = []
current_year = None
current_series = []
for i in range(len(date)):

    year = date[i].astype('datetime64[Y]').astype(int) + 1970
    if current_year is None:
        current_year = year
    if year != current_year:
        time_series_per_year.append(current_series)
        current_series = []
        current_year = year
    current_series.append(data[i])

time_series_per_year.append(current_series)

first_year = date[0].astype('datetime64[Y]').astype(int) + 1970
i_discord = 324454
length = 7 * 24
discord = data[i_discord:i_discord+length]
year_discord = date[i_discord].astype('datetime64[Y]').astype(int) + 1970
year_start = np.where(date == np.datetime64(str(year_discord)))[0][0]
i_year = i_discord -  year_start 
year_pos = year_discord - first_year 
# discord = time_series_per_year[year_pos][i_year:i_year+17] 
print(year_start, date[year_start])
print(date[i_discord])
print(date[i_discord]- date[i_discord].astype('datetime64[M]') + 1)
print(year_discord)

plt.figure()
for i in range(0, len(time_series_per_year), 3):
    plt.plot(time_series_per_year[i], label=f"{first_year + i}")


print(len(time_series_per_year), year_pos)
plt.plot(time_series_per_year[year_pos], label=f"{year_discord}")
plt.plot(np.arange(i_year,i_year+length), discord, color='black', lw=4, label="Discord")
plt.legend()
plt.show()

mp_stomp = load_data("mp_stomp.txt")
smp_stomp = load_data("smp_stomp.txt")
imp_stomp = load_data("imp_stomp.txt")


# plt.plot(data[18400:19000])
# plt.show()
plt.figure()
# plt.plot(smp_bf, label="SMP")
plt.plot(imp_stomp, label="IMP")
plt.plot(mp_stomp, label="MP")
plt.legend()
plt.show(   )



i_mp_3nn = load_data("imp_stomp_kNN.txt")


plt.figure()
plt.plot(i_mp_3nn, label="3NN")
plt.plot(imp_stomp, label="IMP")
plt.legend()
plt.show()