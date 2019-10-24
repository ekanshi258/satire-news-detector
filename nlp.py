import pandas as pd
from imports import bars
from sklearn.feature_extraction import stop_words
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

onion = pd.read_csv('onion.csv')
notonion = pd.read_csv('notonion.csv')

dataf = pd.concat([onion[['subreddit','title']],notonion[['subreddit','title']]], axis = 0)	#concatenating both data
dataf = dataf.reset_index(drop=True)														#into one dataframe	

#binarize
# 1 for TheOnion (false news) , 0 for noththeonion (genuine news)
dataf["subreddit"]=dataf["subreddit"].map({"nottheonion":0, "TheOnion":1})

#Count Vectorization
#------------unigrams------------

#subreddit = 1
vectorizer = CountVectorizer(stop_words = 'english', ngram_range = (1,1))
mask = dataf['subreddit'] == 1
onion_titles = dataf[mask]['title']															#variable for TheOnion titles
#fit vectorizer on the onion titles:
onion_cv = vectorizer.fit_transform(onion_titles)
onion_cv_df = pd.DataFrame(onion_cv.toarray(), columns = vectorizer.get_feature_names())		#convert to dataframe

#subreddit = 0 
vectorizer2 = CountVectorizer(stop_words = 'english', ngram_range = (1,1))
mask2 = dataf['subreddit'] == 0
notonion_titles = dataf[mask2]['title']
notonion_cv = vectorizer2.fit_transform(notonion_titles)
notonion_cv_df = pd.DataFrame(notonion_cv.toarray(), columns = vectorizer2.get_feature_names())

#get unigrams from each 
onion_words = onion_cv_df.sum(axis=0)
onion_top5 = onion_words.sort_values(ascending = False).head(5)
bars(onion_top5.values, onion_top5.index, 'Top 5 TheOnion unigrams','g')

notonion_words = notonion_cv_df.sum(axis = 0)
notonion_top5 = notonion_words.sort_values(ascending=False).head(5)
bars(notonion_top5.values, notonion_top5.index, 'Top 5 nottheonion unigrams', 'b')

notonion_5 = set(notonion_top5.index)
onion_5 = set(onion_top5.index)
common_uni = onion_5.intersection(notonion_5)									#common unigrams between the two

print(common_uni)

#----------bigrams-------------

vectorizer = CountVectorizer(stop_words = 'english', ngram_range = (2,2))
mask = dataf['subreddit'] == 1
onion_titles = dataf[mask]['title']															
onion_cv = vectorizer.fit_transform(onion_titles)
onion_cv_df = pd.DataFrame(onion_cv.toarray(), columns = vectorizer.get_feature_names())		#convert to dataframe

vectorizer2 = CountVectorizer(stop_words = 'english', ngram_range = (2,2))
mask2 = dataf['subreddit'] == 0
notonion_titles = dataf[mask2]['title']
notonion_cv = vectorizer2.fit_transform(notonion_titles)
notonion_cv_df = pd.DataFrame(notonion_cv.toarray(), columns = vectorizer2.get_feature_names())

#get bigrams from each 
onion_words = onion_cv_df.sum(axis=0)
onion_top5 = onion_words.sort_values(ascending = False).head(5)
bars(onion_top5.values, onion_top5.index, 'Top 5 TheOnion bigrams','g')

notonion_words = notonion_cv_df.sum(axis = 0)
notonion_top5 = notonion_words.sort_values(ascending=False).head(5)
bars(notonion_top5.values, notonion_top5.index, 'Top 5 nottheonion bigrams', 'b')

notonion_5 = set(notonion_top5.index)
onion_5 = set(onion_top5.index)
common_bi = onion_5.intersection(notonion_5)	

print(common_bi)