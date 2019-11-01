from imports import bars
import pandas as pd

fake = pd.read_csv('fake.csv')
real = pd.read_csv('real.csv')

#Plot the fake domains vs posts
fake_domain = fake['domain'].value_counts()
fake_domain = fake_domain.sort_values(ascending = False).head(20)			#number of posts per domain
fake_domain_idx = list(fake_domain.index)							#y-axis: domains

bars(fake_domain.values, fake_domain_idx, 'Domains referenced in r/TheOnion','g')

#plot notthefake domains vs posts
real_domain = real['domain'].value_counts()
real_domain = real_domain.sort_values(ascending=False).head(40)
real_domain_idx = list(real_domain.index)

bars(real_domain.values, real_domain_idx, 'Domains referenced in r/NotTheOnion','b')