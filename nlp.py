import pandas as pd
from imports import bars
from sklearn.feature_extraction import stop_words
from sklearn.feature_extraction.text import CountVectorizer

fake = pd.read_csv('clean_data/fake.csv')
real = pd.read_csv('clean_data/real.csv')

dataf = pd.concat([fake[['subreddit','title']],real[['subreddit','title']]], axis = 0)	#concatenating both data
dataf = dataf.reset_index(drop=True)														#into one dataframe	

#binarize
# 1 for TheOnion (false news) , 0 for nothetheonion (genuine news)
dataf["subreddit"]=dataf["subreddit"].map({"nottheonion":0, "TheOnion":1})

# dataf.to_csv("combined.csv") 					#To make "combined.csv"



#Count Vectorization
#------------unigrams------------

#subreddit = 1
vectorizer = CountVectorizer(stop_words = 'english', ngram_range = (1,1))
mask = dataf['subreddit'] == 1
fake_titles = dataf[mask]['title']															#variable for TheOnion titles
fake_cv = vectorizer.fit_transform(fake_titles)
fake_cv_df = pd.DataFrame(fake_cv.toarray(), columns = vectorizer.get_feature_names())		#convert to dataframe

#subreddit = 0 
vectorizer2 = CountVectorizer(stop_words = 'english', ngram_range = (1,1))
mask2 = dataf['subreddit'] == 0
real_titles = dataf[mask2]['title']
real_cv = vectorizer2.fit_transform(real_titles)
real_cv_df = pd.DataFrame(real_cv.toarray(), columns = vectorizer2.get_feature_names())

#get unigrams from each 
fake_words = fake_cv_df.sum(axis=0)
fake_top5 = fake_words.sort_values(ascending = False).head(5)
bars(fake_top5.values, fake_top5.index, 'Top 5 TheOnion unigrams','g')

real_words = real_cv_df.sum(axis = 0)
real_top5 = real_words.sort_values(ascending=False).head(5)
bars(real_top5.values, real_top5.index, 'Top 5 nottheonion unigrams', 'b')

real_5 = set(real_top5.index)
fake_5 = set(fake_top5.index)
common_uni = fake_5.intersection(real_5)									#common unigrams between the two

print(common_uni)

#----------bigrams-------------

vectorizer = CountVectorizer(stop_words = 'english', ngram_range = (2,2))
mask = dataf['subreddit'] == 1
fake_titles = dataf[mask]['title']															
fake_cv = vectorizer.fit_transform(fake_titles)
fake_cv_df = pd.DataFrame(fake_cv.toarray(), columns = vectorizer.get_feature_names())		#convert to dataframe

vectorizer2 = CountVectorizer(stop_words = 'english', ngram_range = (2,2))
mask2 = dataf['subreddit'] == 0
real_titles = dataf[mask2]['title']
real_cv = vectorizer2.fit_transform(real_titles)
real_cv_df = pd.DataFrame(real_cv.toarray(), columns = vectorizer2.get_feature_names())

#get bigrams from each 
fake_words = fake_cv_df.sum(axis=0)
fake_top5 = fake_words.sort_values(ascending = False).head(5)
bars(fake_top5.values, fake_top5.index, 'Top 5 TheOnion bigrams','g')

real_words = real_cv_df.sum(axis = 0)
real_top5 = real_words.sort_values(ascending=False).head(5)
bars(real_top5.values, real_top5.index, 'Top 5 nottheonion bigrams', 'b')

real_5 = set(real_top5.index)
fake_5 = set(fake_top5.index)
common_bi = fake_5.intersection(real_5)	

print(common_bi)

#Stop words list

stop = stop_words.ENGLISH_STOP_WORDS
stop = list(stop)
common_uni = list(common_uni)
common_bi = list(common_bi)
for i in common_uni:
	stop.append(i)
for i in common_bi:
	words = i.split(" ")
	for word in words:
		stop.append(word)

print(stop)