from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
import joblib
import matplotlib.pyplot as plt
import seaborn as sb
import pandas as pd
import numpy as np


fake = pd.read_csv('clean_data/fake.csv')
real = pd.read_csv('clean_data/real.csv')

dataf = pd.concat([fake[['subreddit','title']],real[['subreddit','title']]], axis = 0)	#concatenating both data
dataf = dataf.reset_index(drop=True)														#into one dataframe	
dataf["subreddit"]=dataf["subreddit"].map({"nottheonion":0, "TheOnion":1})

x = dataf['title']
y = dataf['subreddit']
xtrain, xtest, ytrain, ytest = train_test_split(x,y, random_state=42, stratify = y)

mnb = MultinomialNB(alpha = 0.36)
cvec = CountVectorizer(ngram_range = (1,3))

#fit vectorizer
cvec.fit(xtrain)
xcv_train = cvec.transform(xtrain)
xcv_test = cvec.transform(xtest)

#saving the  vectorizer
filename_vectorizer = 'vectorizer.sav'
joblib.dump(cvec, filename_vectorizer)

#fit classifier
mnb.fit(xcv_train, ytrain)

#saving the model
filename_model = 'final_model.sav'
joblib.dump(mnb, filename_model)

#predictions
pred = mnb.predict(xcv_test)

print(mnb.score(xcv_test,ytest))

#-----confusion matrix-------
cnf = metrics.confusion_matrix(ytest,pred)
print(cnf)

# Source: <https://www.datacamp.com/community/tutorials/understanding-logistic-regression-python>
class_names = [0,1]
fig, ax = plt.subplots()
ticks = np.arange(len(class_names))
plt.xticks(ticks, class_names)
plt.yticks(ticks, class_names)

sb.heatmap(pd.DataFrame(cnf), annot = True, cmap = "YlGnBu", fmt = 'g')
#sax.xaxis.set_label_position("top")
#plt.tight_layout()
plt.title('Confusion Matrix', y = 1.1)
plt.ylabel('Actual Subreddit label')
plt.xlabel('Predicted Subreddit label')

plt.show()

cnf = np.array(cnf).tolist()
tnfp, fntp = cnf
fn, tp = fntp
tn, fp = tnfp

print("Percentage Accuracy:",round(metrics.accuracy_score(ytest, pred)*100, 2),'%')
print("Precision:",round(metrics.precision_score(ytest, pred)*100, 2), '%')
print("Recall:",round(metrics.recall_score(ytest, pred)*100, 2), '%')
print("Specificity:", round((tn/(tn+fp))*100, 2), '%')
print("Misclassification:", round((fp+fn)/(tn+fp+fn+tn)*100, 2), '%')