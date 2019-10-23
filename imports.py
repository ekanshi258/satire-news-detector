import numpy as np
import pandas as pd

from sklearn.feature_extraction import stop_words
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

import matplotlib.pyplot as plt
import seaborn as sb

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.metrics import accuracy_score, recall_score, precision_score, confusion_matrix

def clean_data(frame):
	frame.drop_duplicates(subset='title', inplace=True)
	frame['title'] = frame['title'].str.replace('[^\w\s]',' ')			#punctuation removal
	frame['title'] = frame['title'].str.replace('[^A-Za-z]',' ')		#numerals removal
	frame['title'] = frame['title'].str.replace('  ',' ')				#replace double/triple/four spaces with singles
	frame['title'] = frame['title'].str.replace('  ',' ')
	frame['title'] = frame['title'].str.replace('  ',' ')
	frame['title'] = frame['title'].str.lower()							#all text lowercase
	return frame

def bars(x, y, title, colour):
	plt.figure(figsize=(18,20))
	graph = sb.barplot(x, y, color = colour)
	a = graph

	plt.title(title, fontsize = 16)										#labelling the graph
	plt.xticks(fontsize = 8)

	tots = []															#list to collect patches data
	for p in a.patches:
		tots.append(p.get_width())										#values appended to list
	total = sum(tots)

	for p in a.patches:
		a.text(p.get_width()+0.4, p.get_y()+0.4, int(p.get_width()), fontsize=8)	#bar labels
	plt.show()

