import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
def read_and_filter_file(file_path):
    # Read the CSV file into a pandas DataFrame
    df_1500 = pd.read_csv(file_path)
    
    # Filter out rows where 'idx' is zero
    filtered_df = df_1500[df_1500['idx'] != 0.0]
    
    return filtered_df

df_1500 = read_and_filter_file( 'joined_data_1500.txt')
df_900 = read_and_filter_file( 'joined_data_900.txt')

grouped_df = df_1500.groupby('step').mean().reset_index()
df_1500['ftmag'] = np.sqrt((df_1500['ftx']+df_1500['fcx'])**2 + (df_1500['fty']+df_1500['fcy'])**2)
df_1500['ftmag'] = df_1500['ftmag'].round(2)  

grouped_df = df_900.groupby('step').mean().reset_index()
df_900['ftmag'] = np.sqrt((df_900['ftx']+df_900['fcx'])**2 + (df_900['fty']+df_900['fcy'])**2)
df_900['ftmag'] = df_900['ftmag'].round(2)  

# Print the row for df_900 where the column "step" has a value of 5723
row_5723 = df_900[df_900['idx'] == 3484]
print(row_5723)





# Draw histograms of the 'ftmag' column from both DataFrames in one chart
""" plt.hist(df_1500['ftmag'], bins=10, alpha=1, label='1500')
plt.hist(df_900['ftmag'], bins=10, alpha=0.5, label='900', color='r')
plt.title('Histogram of Tangential Force (ftmag)')
plt.xlabel('Tangential Force (ftmag)')
plt.ylabel('Frequency')
plt.show() """

""" print(df_1500['ftmag'].mean())
print(df_1500['ftmag'].max())
print(df_1500['ftmag'].min())
print(len(df_1500))
print(len(grouped_df)) """