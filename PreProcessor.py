import re
import nltk
from unidecode import unidecode
from collections import Counter
from nltk.stem.snowball import SnowballStemmer
from sklearn.base import TransformerMixin, BaseEstimator
import logging


class Preprocessor(TransformerMixin, BaseEstimator):
    
    def __init__(self, lang='english', stop_words=True, stem=False):
        self.lang = lang                
        self.stop_words = stop_words
        self.stem = stem        
     
    def cleaning(self, corpus):
        return [self.strip_accents_nonalpha(text) for text in corpus]
        
    
    def strip_accents_nonalpha(self, text):
        text = text.lower()
        t = unidecode(text)
        t.encode("ascii")
        t = re.sub(r'[^a-z]', ' ', t)       
        t = ' '.join(t.strip().split())
        return t
    
    def stem_text(self, text):
        stemmer = SnowballStemmer(self.lang)
        return [stemmer.stem(x) for x in text]

    def remove_stopwords(self, corpus):
        stopwords = nltk.corpus.stopwords.words(self.lang)
        stopwords = [self.strip_accents_nonalpha(x) for x in stopwords]
        pattern = re.compile(r'\b(' + r'|'.join(stopwords) + r')\b\s*')
        return [pattern.sub('', text) for text in corpus]
    

    def preprocess(self, corpus):        
        corpus = self.cleaning(corpus)
        if(self.stop_words):
            corpus = self.remove_stopwords(corpus)
        corpus = [x.split() for x in corpus]        
        if(self.stem):
            corpus = [self.stem_text(x) for x in corpus]
        return corpus
    
    def fit(self, X, y=None):                
        return self
    
    def transform(self, X, *_):
        return self.preprocess(X)
            
    
    
    