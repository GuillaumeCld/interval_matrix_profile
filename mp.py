import pandas as pd
import stumpy 
import numpy as np
import matplotlib.pyplot as plt 
import plotly.graph_objs as go

df = pd.read_csv("Data/era5_daily_series_sst_60S-60N_ocean.csv", sep=",", skiprows=18)
df = df[df["status"] == "FINAL"]
ts = np.array(df["sst"]).astype(np.double)


stumpy_out = stumpy.stump(ts, m=7, normalize=False)
mp = stumpy_out[:,0].astype(np.double)

with open("Data/ts_sst.txt", "w") as file:
    for elem in ts:
        file.write(f"{elem:.2f}\n")

with open("Data/mp_sst.txt", "w") as file:
    for elem in mp:
        file.write(f"{elem:.7f}\n")



mp_bf = np.zeros_like(mp)
mp_stomp = np.zeros_like(mp)

ind_bf = np.zeros(len(mp), dtype=int)
ind_stomp = np.zeros(len(mp), dtype=int)

with open("Data/matrix_profile_bf.txt", "r") as file:
    for i, line in enumerate(file):
        mp_bf[i] = float(line)


with open("Data/matrix_profile_stomp.txt", "r") as file:
    for i, line in enumerate(file):
        mp_stomp[i] = float(line)   

with open("Data/index_profile_bf.txt", "r") as file:
    for i, line in enumerate(file):
        ind_bf[i] = int(line)


with open("Data/index_profile_stomp.txt", "r") as file:
    for i, line in enumerate(file):
        ind_stomp[i] = int(line)   


print("Array equal to STUMP with BF: ", np.allclose(mp, mp_bf))
print("Array equal to STUMP with STOM: ", np.allclose(mp, mp_stomp))



print("Dist ", np.linalg.norm(ts[4038:4038+7]-ts[5489:5489+7]))
print("Dist ", np.linalg.norm(ts[4038:4038+7]-ts[0:0+7]))

plt.figure()
plt.plot(mp, label="STUMP") 
plt.plot(mp_bf, label="BF")
plt.plot(mp_stomp, label="STOMP")
plt.legend()    
plt.savefig("Data/mp_sst.png")


plt.figure()
plt.plot(mp-mp_bf)    
plt.savefig("Data/diff_bf.png")

plt.figure()
plt.plot(mp-mp_stomp)    
plt.savefig("Data/diff_stomp.png")



plt.figure()
plt.plot(mp_bf-mp_bf)    
plt.savefig("Data/diff_bf_stomp.png")

hover_text_1 = [f"{i}" for i in stumpy_out[:,1]]
hover_text_2 = [f"{i}" for i in ind_bf]
hover_text_3 = [f"{i}" for i in ind_stomp]
# Create traces
trace1 = go.Scatter(y=mp, mode='lines', name='STUMP', text=hover_text_1)
trace2 = go.Scatter(y=mp_bf, mode='lines', name='BF', text=hover_text_2)
trace3 = go.Scatter(y=mp_stomp, mode='lines', name='STOMP', text=hover_text_3)

# Layout
layout = go.Layout(xaxis=dict(title='Distance'))

# Plot
fig = go.Figure(data=[trace1, trace2, trace3], layout=layout)
fig.show()