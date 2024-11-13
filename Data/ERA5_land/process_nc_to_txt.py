import xarray as xr 
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


def build_four_seasons_idx_hourly(ts_date, start_year, end_year, seasons_date):
    years = range(start_year, end_year+1)

    seasons = [ np.zeros(2*len(years), dtype=int),  np.zeros(2*len(years), dtype=int),  np.zeros(2*len(years), dtype=int),  np.zeros(2*len(years) + 2, dtype=int) ]
   
    spring, summer, automn, winter = seasons_date

    
    for i,year_int in enumerate(years):
        dyear = str(year_int) + "-"
        first_hour = "T00"
        
        seasons[0][2*i] = np.where(ts_date == np.datetime64(dyear+spring+first_hour))[0][0]
        seasons[0][2*i+1] = np.where(ts_date == np.datetime64(dyear+summer+first_hour))[0][0]
        
        seasons[1][2*i] = np.where(ts_date == np.datetime64(dyear+summer+first_hour))[0][0]
        seasons[1][2*i+1] = np.where(ts_date == np.datetime64(dyear+automn+first_hour))[0][0]
       
        seasons[2][2*i] = np.where(ts_date == np.datetime64(dyear+automn+first_hour))[0][0]
        seasons[2][2*i+1] = np.where(ts_date == np.datetime64(dyear+winter+first_hour))[0][0]
       
        
        
        nyear = str(year_int+1) + "-"
        if year_int == start_year :
            seasons[3][0] = np.where(ts_date == np.datetime64(f"{start_year}-01-01T01"))[0][0]
            seasons[3][1] = np.where(ts_date == np.datetime64(dyear+spring+first_hour))[0][0]
            seasons[3][2 + 2*i] = np.where(ts_date == np.datetime64(dyear+winter+first_hour))[0][0]
            seasons[3][2 + 2*i+1] = np.where(ts_date == np.datetime64(nyear+spring+first_hour))[0][0]
        
        elif year_int == end_year :
            seasons[3][2 + 2*i] = np.where(ts_date == np.datetime64(dyear+winter+first_hour))[0][0]
            seasons[3][2 + 2*i+1] = np.where(ts_date == np.datetime64(dyear+'12-31T23'))[0][0]

        else :
            
            seasons[3][2 + 2*i] = np.where(ts_date == np.datetime64(dyear+winter+first_hour))[0][0]
            seasons[3][2 + 2*i+1] = np.where(ts_date == np.datetime64(nyear+spring+first_hour))[0][0]
                
    return seasons


def build_daily(filename):
    ds = xr.open_dataset(filename)
    start_year = 1950
    end_year = 2023
    ts = ds.t2m.data.astype(np.float64)
    ts_date = ds.time.data.astype('datetime64[D]')

    # Save the time series
    with open("daily_series.txt", "w") as file:
        for elem in ts:
            file.write(f"{elem}\n")

    # Build the four seasons index for SMP
    spring = '03-21'
    summer = '06-21'
    automn = '09-21'
    winter = '12-21'
    four_seaons = [spring, summer, automn, winter]
    seasons = build_four_seasons_idx(ts_date,  start_year, end_year, four_seaons)

    for i in range(4):
        with open(f"seasons_daily_{i}.txt", "w") as file:
            for elem in seasons[i]:
                file.write(f"{elem}\n")

    # Build the start of the year index for IMP
    list_ind = []
    for year in range(start_year, end_year+1):
        list_ind.append(np.where(ts_date == np.datetime64(str(year) + "-01-01"))[0][0])

    with open("periods_start_daily.txt", "w") as f:
        for item in list_ind:
            f.write(f"{item}\n")

def build_hourly(filename):
    ds = xr.open_dataset(filename)
    start_year = 1950
    end_year = 2023
    ts = ds.t2m.data.astype(np.float64)
    ts_date = ds.time.data.astype('datetime64[h]')

    # Save the time series
    with open("hourly_series.txt", "w") as file:
        for elem in ts:
            file.write(f"{elem}\n")

    # Build the four seasons index for SMP
    spring = '03-21'
    summer = '06-21'
    automn = '09-21'
    winter = '12-21'
    four_seaons = [spring, summer, automn, winter]
    seasons = build_four_seasons_idx_hourly(ts_date,  start_year, end_year, four_seaons)

    for i in range(4):
        with open(f"seasons_hourly_{i}.txt", "w") as file:
            for elem in seasons[i]:
                file.write(f"{elem}\n")

    # Build the start/end of the year index for IMP
    list_ind = []
    for year in range(start_year, end_year+1):
        if year == 1950:
            list_ind.append(np.where(ts_date == np.datetime64(str(year) + "-01-01T01"))[0][0])
        else:
            list_ind.append(np.where(ts_date == np.datetime64(str(year) + "-01-01T00"))[0][0])

    with open("periods_start_hourly.txt", "w") as f:
        for item in list_ind:
            f.write(f"{item}\n")

if __name__ == "__main__":
    build_daily("Bordeaux_t2m_daily_avg_1950_2023.nc")
    build_hourly("Bordeaux_t2m_hourly_avg_1950_2023.nc")