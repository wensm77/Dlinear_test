import pandas as pd

# 合并 csv
for i in range(4):
    df = pd.read_csv(f'data_{i+1}.csv')
    if i == 0:
        df_all = df
    else:
        df_all = pd.concat([df_all, df], ignore_index=True)

df_all.to_csv('data_all.csv', index=False)