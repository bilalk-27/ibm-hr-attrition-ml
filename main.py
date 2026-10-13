import pandas as pd

df = pd.read_csv('kaggle_ibm_dataset_attriction.csv')
print(df.isna().sum())

