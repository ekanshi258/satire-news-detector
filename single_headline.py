from sklearn.feature_extraction.text import CountVectorizer
import joblib
import pandas as pd

#Placeholder headline
#Will be read from user input in final version
filename = 'final_model.sav'
loaded_model = joblib.load(filename)

headline = "Poll Shows Support For Impeachment Weakest Among Uncontacted Amazonian Tribes Who Know Nothing Of Our Ways"
headline = headline.replace('[^\w\s]',' ')			#punctuation removal
headline = headline.replace('[^A-Za-z]',' ')		#numerals removal
headline = headline.replace('  ',' ')				#replace double/triple/four spaces with singles
headline = headline.replace('  ',' ')
headline = headline.replace('  ',' ')
headline = headline.lower()				

cvec = CountVectorizer(ngram_range = (1,3), stop_words = 'english')

headline = pd.DataFrame([headline], columns = ['Title'])

cvec.fit(headline)
cv = cvec.transform(headline)

#this doesn't work
#needs to be fixed
result = loaded_model.predict(cv)

print(result)