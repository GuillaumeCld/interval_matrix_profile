import pandas as pd
import stumpy 

df = pd.read_csv("Data/era5_daily_series_sst_60S-60N_ocean.csv", sep=",", skiprows=18)
df = df[df["status"] == "FINAL"]
ts = df["sst"]


stumpy_out = stumpy.stump(ts, m=7, normalize=False)
mp = stumpy_out[:,0]

with open("Data/ts_sst.txt", "w") as file:
    for elem in ts:
        file.write(f"{elem:.2f}\n")

with open("Data/mp_sst.txt", "w") as file:
    for elem in mp:
        file.write(f"{elem}\n")