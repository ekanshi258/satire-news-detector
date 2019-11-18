import pandas as pd
from imports import clean_data

satire= pd.read_csv('./data/satire.csv')
real = pd.read_csv('./data/satire.csv')

df_satire = clean_data(satire)
df_satire.to_csv('satire.csv')
df_real = clean_data(real)
df_real.to_csv('real.csv')