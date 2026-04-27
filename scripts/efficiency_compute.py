import pandas as pd


data = pd.read_csv('output_files/imp_parallel.txt', sep=",", header=None)
data.columns = ["Cores", "Time", "Speedup"]

# compute the efficiency
data['Efficiency'] = data['Speedup'] / data['Cores']

print(f"Min efficiency: {data['Efficiency'].min()}")
print(f"Max efficiency: {data['Efficiency'].max()}")
print(f"Mean efficiency: {data['Efficiency'].mean()}")
print(f"Median efficiency: {data['Efficiency'].median()}")

print(data)

