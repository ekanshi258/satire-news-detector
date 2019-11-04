# fake-news-detector
Made for an Assignment for the Artificial Intelligence course (CS F407) at BITS Pilani, Hyderabad.
### Files and Folders
`barplots/`: Bar plots of useful data from real and fake news datasets (scraped from online sources)  
`data/`: .csv files of data
`clean_data/`: .csv files of cleaned data
`frontend/` : Contains the User-Interface for the website, developed in Python Flask.
`imports.py`: functions and library imports  
`dataclean.py`: program to clean the raw data from `data/`  
`domains.py`: program to find most common domains referenced in both datasets
`combined.csv`: Combined clean data of both datasets. Further work wil be carried out using this data.  
`nlp.py`: NLP on the clean, combined data  
`Text_Outputs.txt`: Intermediate outputs the NLP process, for reference.
`model.py` : Working model which uses a Naive Bayes Classifier at its core.
`optimal.py` : GridSearch parameters on two different models for determining the best model and its parameters.
# Group
Ekanshi Agrawal - [@ekanshi258](https://github.com/ekanshi258)  
Kushagra Srivastava - [@z3r0dmg](https://github.com/z3r0dmg)  
Kunal Verma - [@stumblef00l](https://github.com/stumblef00l)  
