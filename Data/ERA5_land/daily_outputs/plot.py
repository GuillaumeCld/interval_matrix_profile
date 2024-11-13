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

ds = xr.open_dataset("../Bordeaux_t2m_daily_avg_1950_2023.nc")
date = ds.time.values.astype('datetime64[D]')
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
i_discord = 18431
discord = data[i_discord:i_discord+7]
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
plt.plot(np.arange(i_year,i_year+7), discord, color='black', lw=4, label="Discord")
plt.legend()
plt.show()

mp_bf = load_data("mp_bf.txt")
mp_stomp = load_data("mp_stomp.txt")
smp_bf = load_data("smp_bf.txt")
smp_stomp = load_data("smp_stomp.txt")
imp_bf = load_data("imp_bf.txt")
imp_stomp = load_data("imp_stomp.txt")
mp_bf_ind = load_data("mp_bf_index.txt").astype(int)
mp_stomp_ind = load_data("mp_stomp_index.txt").astype(int)
smp_bf_ind = load_data("smp_bf_index.txt").astype(int)
smp_stomp_ind = load_data("smp_stomp_index.txt").astype(int)
imp_bf_ind = load_data("imp_bf_index.txt").astype(int)
imp_stomp_ind = load_data("imp_stomp_index.txt").astype(int)


print("Lenght ", len(mp_bf))
assert(mp_bf.shape == mp_stomp.shape)
assert(smp_bf.shape == smp_stomp.shape)
assert(imp_bf.shape == imp_stomp.shape)

for i, (bf, stomp) in enumerate(zip(imp_bf, imp_stomp)):
    seq = data[i:i+7]
    
    i_bf = imp_bf_ind[i]
    seq_bf = data[i_bf:i_bf+7]
    
    i_stomp = imp_stomp_ind[i]
    seq_stomp = data[i_stomp:i_stomp+7]
    # assert (i_bf == i_stomp), f"Index mismatch, {i_bf} != {i_stomp}, i={i}"
    bf_value = np.sqrt(np.sum((seq - seq_bf)**2))   
    stomp_value = np.sqrt(np.sum((seq - seq_stomp)**2))
    
    assert np.isclose(bf, stomp), f"For IMP BF!=STOMP, {bf} != {stomp}, i={i} at index {imp_bf_ind[i]}, {imp_stomp_ind[i]},\
          {bf_value}, {stomp_value} \n  \
            {i%365} {imp_bf_ind[i]%365}, {imp_stomp_ind[i]%365}\n\
            {imp_bf_ind[i]%365 - i%365}, {imp_stomp_ind[i]%365 - i%365}"


for i, (bf, stomp) in enumerate(zip(smp_bf, smp_stomp)):
    assert np.isclose(bf, stomp), f"For SMP BF!=STOMP, {bf} != {stomp}, i={i}, at index {smp_bf_ind[i]}, {smp_stomp_ind[i]}"

for i, (bf, stomp) in enumerate(zip(mp_bf, mp_stomp)):
    assert np.isclose(bf, stomp), f"For MP BF!=STOMP, {bf} != {stomp}, i={i}, at index {mp_bf_ind[i]}, {mp_stomp_ind[i]} "


# plt.plot(data[18400:19000])
# plt.show()
plt.figure()
# plt.plot(smp_bf, label="SMP")
plt.plot(imp_bf, label="IMP")
plt.plot(mp_bf, label="MP")
plt.legend()
plt.show(   )



i_mp_3nn = load_data("imp_stomp_kNN.txt")


plt.figure()
plt.plot(i_mp_3nn, label="3NN")
plt.plot(imp_stomp, label="IMP")
plt.legend()
plt.show()