import numpy as np
import pandas as pd
from imports import clean_data

onion = pd.read_csv('./data/the_onion.csv')
notonion = pd.read_csv('./data/not_onion.csv')

df_onion = clean_data(onion)
df_onion.to_csv('onion.csv')
df_notonion = clean_data(notonion)
df_notonion.to_csv('notonion.csv')