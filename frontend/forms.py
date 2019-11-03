from flask_wtf import Form
from wtforms import StringField, SubmitField
from wtforms.validators import DataRequired

class SearchForm(Form):
	query = StringField('Enter query',validators=[DataRequired()])
	submit = SubmitField('Search')