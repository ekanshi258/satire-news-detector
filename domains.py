from imports import bars
import pandas as pd

satire = pd.read_csv('./clean_data/satire.csv')
real = pd.read_csv('./clean_data/real.csv')

#Plot the satire domains vs posts
satire_domain = satire['domain'].value_counts()
satire_domain = satire_domain.sort_values(ascending = False).head(10)			#number of posts per domain
satire_domain_idx = list(satire_domain.index)							#y-axis: domains

bars(satire_domain.values, satire_domain_idx, 'Domains referenced in satire news dataset','g')

#plot notthesatire domains vs posts
real_domain = real['domain'].value_counts()
real_domain = real_domain.sort_values(ascending=False).head(10)
real_domain_idx = list(real_domain.index)

bars(real_domain.values, real_domain_idx, 'Domains referenced in real news dataset','b')