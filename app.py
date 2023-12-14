import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize
import os
import nltk

tfidf = pickle.load(open('tfidfv', 'rb'))
model=pickle.load(open('mnbc','rb'))
nltk.download('stopwords')
nltk.download('punkt')
st.title('SMS Spam Classifier')
sms=st.text_area(label='sms input')

def vector_text(text):
    text=text.lower()
    text=text.translate(str.maketrans('','',string.punctuation))
    ps=PorterStemmer()
    text=' '.join([ps.stem(word) for word in word_tokenize(text) if word not in stopwords.words('english')])
    return text
if st.button('Know Spam sms'):
    cleaned_sms=vector_text(sms)
    vector_sms=tfidf.transform([cleaned_sms])
    pred=model.predict(vector_sms)[0]
    if pred==1:
        st.header('Spam')
    else:
        st.header('Not Spam')
