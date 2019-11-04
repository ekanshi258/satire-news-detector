from flask import Flask, render_template, url_for, request
from forms import SearchForm
import joblib

app = Flask(__name__)

app.config['SECRET_KEY'] = '6a5c6dc14729082181c6266a6adeabd0'

def getPrediction(headline):
    
    filename_model = 'final_model.sav'
    filename_vectorizer = 'vectorizer.sav'
    loaded_model = joblib.load(filename_model)
    loaded_vectorizer = joblib.load(filename_vectorizer)

    headline = headline.replace('[^\w\s]',' ')			#punctuation removal
    headline = headline.replace('[^A-Za-z]',' ')		#numerals removal
    headline = headline.replace('  ',' ')				#replace double/triple/four spaces with singles
    headline = headline.replace('  ',' ')
    headline = headline.replace('  ',' ')
    headline = headline.lower()
    
    cv = loaded_vectorizer.transform([headline])
    
    result = loaded_model.predict(cv)
    
    if(result[0] == 0):
        return ["real"]
    else:
        return ["fake"]

#Main page
@app.route("/", methods=['GET', 'POST'])
def home():
	form = SearchForm(request.form)
	if request.method == 'POST' and form.validate():
		query = form.query.data
		print(query)
		results = getPrediction(str(query))
		return render_template('results.html',results=results, query=query)

	return render_template('search.html',form=form,res=True)


if __name__ == '__main__':
	app.run()