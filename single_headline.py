import joblib
#Placeholder headline
#Will be read from user input in final version
filename_model = 'final_model.sav'
filename_vectorizer = 'vectorizer.sav'
loaded_model = joblib.load(filename_model)
loaded_vectorizer = joblib.load(filename_vectorizer)

def getPredictions(headline):
    
    headline = "surf excel faces backlash for ad promoting hindu muslim harmony times of india"
    headline = headline.replace('[^\w\s]',' ')			#punctuation removal
    headline = headline.replace('[^A-Za-z]',' ')		#numerals removal
    headline = headline.replace('  ',' ')				#replace double/triple/four spaces with singles
    headline = headline.replace('  ',' ')
    headline = headline.replace('  ',' ')
    headline = headline.lower()
    
    cv = loaded_vectorizer.transform([headline])
    
    result = loaded_model.predict(cv)
    
    if(result[0] == 0):
        return "real"
    else:
        return "fake"
    
    
