import pandas as pd



ds_bf = pd.read_csv('output_files/imp_interval_impact_bf.txt', sep=',')
ds_aamp = pd.read_csv('output_files/imp_interval_impact_aamp.txt', sep=',')
ds_block = pd.read_csv('output_files/imp_interval_impact_block.txt', sep=',')


bf_slope = ds_bf['Time'].diff().mean()
aamp_slope = ds_aamp['Time'].diff().mean()
block_slope = ds_block['Time'].diff().mean()

print("Slope of the impact interval graph")
print(f"BF: {bf_slope:.2f}")
print(f"AAMP: {aamp_slope:.2f}")
print(f"Block: {block_slope:.2f}")