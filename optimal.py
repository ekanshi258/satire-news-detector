from nlp import dataf
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer

#baseline
print(dataf['subreddit'].value_counts(normalize = True))

#Predictor (x) and target (y)
x = dataf['title']
y = dataf['subreddit']
xtrain, xtest, ytrain, ytest = train_test_split(x,y, random_state=42, stratify = y)

"""
    
    Finding best parameters for Multinomial Naive Bayes Classifier
    
"""


#GridSearchCV
pipe = Pipeline([('vect', CountVectorizer()), ('nb', MultinomialNB())])
params = {'vect__ngram_range':[(1,1),(1,3)], 'nb__alpha':[0.36, 0.6]}

gs = GridSearchCV(pipe, param_grid = params, cv = 3)
gs.fit(xtrain,ytrain)

print("best score: ",gs.best_score_)
print("train score: ", gs.score(xtrain,ytrain))
print("test score: ",gs.score(xtest,ytest))

print(gs.best_params_)

"""

    Find best parameters  for Logistic Regression
    
"""

#GridSearchCV
pipe = Pipeline([('vect', CountVectorizer()), ('lr', LogisticRegression(solver = 'liblinear'))])
params = {'vect__ngram_range':[(1,1),(1,3)], 'lr__C':[0.01, 1]}

gs = GridSearchCV(pipe, param_grid = params, cv = 3)
gs.fit(xtrain,ytrain)

print("best score: ",gs.best_score_)
print("train score: ", gs.score(xtrain,ytrain))
print("test score: ",gs.score(xtest,ytest))

print(gs.best_params_)