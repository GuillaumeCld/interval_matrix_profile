import pandas as pd
import stumpy 
import numpy as np
import matplotlib.pyplot as plt 
import plotly.graph_objs as go

def build_four_seasons_idx(ts_date, start_year, end_year, seasons_date):
    years = range(start_year, end_year+1)

    seasons = [ np.zeros(2*len(years), dtype=int),  np.zeros(2*len(years), dtype=int),  np.zeros(2*len(years), dtype=int),  np.zeros(2*len(years) + 2, dtype=int) ]
   
    spring, summer, automn, winter = seasons_date

    
    for i,year_int in enumerate(years):
       
        dyear = str(year_int) + "-"
        
        
        seasons[0][2*i] = np.where(ts_date == np.datetime64(dyear+spring))[0][0]
        seasons[0][2*i+1] = np.where(ts_date == np.datetime64(dyear+summer))[0][0]
        
        seasons[1][2*i] = np.where(ts_date == np.datetime64(dyear+summer))[0][0]
        seasons[1][2*i+1] = np.where(ts_date == np.datetime64(dyear+automn))[0][0]
       
        seasons[2][2*i] = np.where(ts_date == np.datetime64(dyear+automn))[0][0]
        seasons[2][2*i+1] = np.where(ts_date == np.datetime64(dyear+winter))[0][0]
       
        
        
        nyear = str(year_int+1) + "-"
        if year_int == start_year :
            seasons[3][0] = np.where(ts_date == np.datetime64(f"{start_year}-01-01"))[0][0]
            seasons[3][1] = np.where(ts_date == np.datetime64(dyear+spring))[0][0]
            seasons[3][2 + 2*i] = np.where(ts_date == np.datetime64(dyear+winter))[0][0]
            seasons[3][2 + 2*i+1] = np.where(ts_date == np.datetime64(nyear+spring))[0][0]
        
        elif year_int == end_year :
            seasons[3][2 + 2*i] = np.where(ts_date == np.datetime64(dyear+winter))[0][0]
            seasons[3][2 + 2*i+1] = np.where(ts_date == np.datetime64(dyear+'12-31'))[0][0]

        else :
            seasons[3][2 + 2*i] = np.where(ts_date == np.datetime64(dyear+winter))[0][0]
            seasons[3][2 + 2*i+1] = np.where(ts_date == np.datetime64(nyear+spring))[0][0]
                
    return seasons

df = pd.read_csv("Data/era5_daily_series_sst_60S-60N_ocean.csv", sep=",", skiprows=18)
df = df[df["status"] == "FINAL"]
df = df[(df['date'] >= '1940-01-01') & (df['date'] <= '2023-12-31')]
ts = np.array(df["sst"]).astype(np.double)

ts_date = df["date"].values.astype('datetime64[D]')
spring = '03-21'
summer = '06-21'
automn = '09-21'
winter = '12-21'
four_seaons = [spring, summer, automn, winter]
seasons = build_four_seasons_idx(ts_date,  1979, 2023, four_seaons)

for i in range(4):
    with open(f"Data/seasons_sst_{i}.txt", "w") as file:
        for elem in seasons[i]:
            file.write(f"{elem}\n")
                       
stumpy_out = stumpy.stump(ts, m=7, normalize=False)
mp = stumpy_out[:,0].astype(np.double)

with open("Data/ts_sst.txt", "w") as file:
    for elem in ts:
        file.write(f"{elem:.2f}\n")

with open("Data/mp_sst.txt", "w") as file:
    for elem in mp:
        file.write(f"{elem:.7f}\n")

smp = np.load("Data/smp_sst.npy")
with open("Data/smp_sst.txt", "w") as file:
    for elem in smp:
        file.write(f"{elem:.7f}\n")


smp_ind = np.load("Data/smp_ind_sst.npy")
with open("Data/smp_ind_sst.txt", "w") as file:
    for elem in smp_ind:
        file.write(f"{elem}\n")