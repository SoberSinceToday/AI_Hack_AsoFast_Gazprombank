import pandas as pd

df1 = pd.read_csv('test_predict1.csv')
df2 = pd.read_csv('test_predict2.csv')
df3 = pd.read_csv('test_predict3.csv')
merged_df = df1.merge(df2, on='id', how='outer').merge(df3, on='id', how='outer')
merged_df.to_parquet('ans.parquet')