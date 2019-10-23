from imports import bars
import pandas as pd

onion = pd.read_csv('onion.csv')
notonion = pd.read_csv('notonion.csv')

#Plot the onion domains vs posts
onion_domain = onion['domain'].value_counts()
onion_domain = onion_domain.sort_values(ascending = False).head(20)			#number of posts per domain
onion_domain_idx = list(onion_domain.index)							#y-axis: domains

bars(onion_domain.values, onion_domain_idx, 'Domains referenced in r/TheOnion','g')

#plot nottheonion domains vs posts
notonion_domain = notonion['domain'].value_counts()
notonion_domain = notonion_domain.sort_values(ascending=False).head(40)
notonion_domain_idx = list(notonion_domain.index)

bars(notonion_domain.values, notonion_domain_idx, 'Domains referenced in r/NotTheOnion','b')