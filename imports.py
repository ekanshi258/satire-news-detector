import numpy as np
import pandas as pd
from sklearn.feature_extraction import stop_words
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# Modeling
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
