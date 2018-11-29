import os
import numpy as np
from scipy.spatial.distance import cosine
from gensim.models.wrappers import LdaMallet
from gensim import corpora
from gensim.corpora.dictionary import Dictionary

class DocumentRetriever:
    def __init__(self, path):
        self.path = path
        arr = []
        lines = open(path, "r").read().splitlines()
        for line in lines:
            arr.append(line.split()[2:])
        self.matrix = np.array(arr, dtype=np.float64)
    
    def n_most_similar(self, doc, n=3, metric='cosine'):
        result = []
        if (metric=='cosine'):
            distances = []
            i = 0;
            for item in self.matrix:
                distances.append([i,cosine(doc, item)])
                i+=1
            distances.sort(key=lambda x: x[1])  
            for rank in distances[:n]:
                result.append(rank[0])
        return result
    
    def doc_topics(self, doc_idx, sorted=True):
        topics = []
        i = 0;
        for topic in self.matrix[doc_idx]:
            topics.append([i,topic])
            i+=1
        topics.sort(key=lambda x: x[1], reverse=True)
        return topics
        

class LdaMalletHandler:
    def __init__(self, mallet_path):
        self.mallet_path = mallet_path

    def run_model(self, model_name, corpus, **kwargs):
        self.model_name = model_name
        self.dictionary = Dictionary(corpus)
        corpus_bow = [self.dictionary.doc2bow(text) for text in corpus]
        os.makedirs("ldamodels/"+model_name, exist_ok=True )
        self.model = LdaMallet(self.mallet_path, corpus_bow, id2word=self.dictionary, prefix="./ldamodels/"+model_name+"/", **kwargs)

    def save_model(self):
        self.model.save("ldamodels/"+self.model_name+"/model.model")
        self.dictionary.save("ldamodels/"+self.model_name+"/dict.dict")

    def load_model(self, model_name):
        self.model_name = model_name
        self.dictionary  = corpora.Dictionary.load("ldamodels/"+self.model_name+"/dict.dict")
        self.model = LdaMallet.load("ldamodels/"+self.model_name+"/model.model")
        self.model.mallet_path = self.mallet_path
    
    def doc_topics(self, doc_idx):
        if(not hasattr(self, 'doc_retriever')):
            self.doc_retriever =  DocumentRetriever(self.model.fdoctopics())
        return self.doc_retriever.doc_topics(doc_idx)    
    
    def ext_doc_topics(self, ext_doc):
        doc_bow = self.dictionary.doc2bow(ext_doc)
        doc_topics = self.model[doc_bow]
        doc_topics.sort(key=lambda x: x[1], reverse=True)
        return doc_topics

    def ext_doc_n_most_similar(self, ext_doc, n=5, metric='cosine'):
        if(not hasattr(self, 'doc_retriever')):
            self.doc_retriever =  DocumentRetriever(self.model.fdoctopics())
        doc_bow = self.dictionary.doc2bow(ext_doc)
        doc_topics = self.model[doc_bow]
        topics = []
        for topic in doc_topics:
            topics.append(topic[1])    
        most_similar = self.doc_retriever.n_most_similar(topics, n=n, metric=metric)    
        return most_similar

    def n_most_representative(self, topic, n=3):
         if(not hasattr(self, 'doc_retriever')):
            self.doc_retriever =  DocumentRetriever(self.model.fdoctopics())
         topics = np.zeros(self.model.num_topics)
         topics[topic]=1
         most_similar = self.doc_retriever.n_most_similar(topics, n=n)
         return most_similar
        
    def get_string_topics(self, num_topics=-1, num_words=10):
        if(num_topics==-1):
            num_topics = self.model.num_topics 
        string_topics = []
        for topic in self.model.print_topics(num_topics=num_topics, num_words=num_words):
            splitted = topic[1].split("\"")
            result = [splitted[2*i+1] for i in range(0,int(len(splitted)/2))]
            string_topics.append(" ".join(result))
        return string_topics    
