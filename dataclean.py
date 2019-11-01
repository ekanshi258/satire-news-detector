import pandas as pd
from imports import clean_data

fake= pd.read_csv('./data/fake.csv')
real = pd.read_csv('./data/fake.csv')

df_fake = clean_data(fake)
df_fake.to_csv('fake.csv')
df_real = clean_data(real)
df_real.to_csv('real.csv')