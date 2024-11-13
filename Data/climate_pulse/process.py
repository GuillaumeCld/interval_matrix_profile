import numpy as np
import pandas as pd

# SST
df = pd.read_csv("SST/era5_daily_series_sst_60S-60N_ocean.csv", sep=",", skiprows=18)
df = df[df["status"] == "FINAL"]
start_year = 1979
end_year = 2023
df = df[(df['date'] >= f'{start_year}-01-01') & (df['date'] <= f'{end_year}-12-31')]
ts = np.array(df["sst"]).astype(np.double)
ts_date = df["date"].values.astype('datetime64[D]')

np.save("SST/daily_series.npy", ts)
np.save("SST/daily_series_date.npy", ts_date)
with open("SST/daily_series.txt", "w") as file:
    for elem in ts:
        file.write(f"{elem}\n")

# Build the start of the year index for IMP
list_ind = []
for year in range(start_year, end_year+1):
    list_ind.append(np.where(ts_date == np.datetime64(str(year) + "-01-01"))[0][0])

with open("SST/periods_start_daily.txt", "w") as f:
    for item in list_ind:
        f.write(f"{item}\n")


# T2M

df = pd.read_csv("T2M/era5_daily_series_2t_global.csv", sep=",", skiprows=18)
df = df[df["status"] == "FINAL"]
start_year = 1940
end_year = 2023
df = df[(df['date'] >= f'{start_year}-01-01') & (df['date'] <= f'{end_year}-12-31')]
ts = np.array(df["2t"]).astype(np.double)
ts_date = df["date"].values.astype('datetime64[D]')
np.save("T2M/daily_series.npy", ts)
np.save("T2M/daily_series_date.npy", ts_date)
with open("T2M/daily_series.txt", "w") as file:
    for elem in ts:
        file.write(f"{elem}\n")

# Build the start of the year index for IMP
list_ind = []
for year in range(start_year, end_year+1):
    list_ind.append(np.where(ts_date == np.datetime64(str(year) + "-01-01"))[0][0])

with open("T2M/periods_start_daily.txt", "w") as f:
    for item in list_ind:
        f.write(f"{item}\n")