import gensim
import os
import numpy as np
from nltk.stem.snowball import SnowballStemmer
from scipy.spatial import distance
stm = SnowballStemmer("portuguese")
import warnings
warnings.filterwarnings("ignore")

def combination(word_list):
        comb_list = []
        for i in range(len(word_list)-1):
            for j in range(i+1, len(word_list)):
                if(word_list[i]<word_list[j]):
                    comb_list.append([word_list[i], word_list[j]])
                else:
                    comb_list.append([word_list[j], word_list[i]])
        return comb_list

class Word2Vec_Evaluation:
    def __init__(self, model_path, stem):
        self.stem = stem
        self.model = gensim.models.Word2Vec.load(model_path)

    def pair_coherence(self, word_i, word_j, metric=None):
        if(metric=="correlation"):
            return 1 - distance.correlation(self.model[word_i], self.model[word_j])
        if(metric=="chebyshev"):
            return 1 - distance.chebyshev(self.model[word_i], self.model[word_j])
        if(metric=="euclidean"):
            return 1 - distance.euclidean(self.model[word_i], self.model[word_j])
        if(metric=="canberra"):
            return 1 - distance.canberra(self.model[word_i], self.model[word_j])
        return self.model.similarity(word_i,word_j)

    def get_valid_words(self, topic):
        words = topic.split()
        new_list = []
        self.rev_dic = {}
        
        if(self.stem):
            for word in words:
                if(stm.stem(word) in self.model.wv.vocab.keys()):
                    self.rev_dic[stm.stem(word)] = word 
                    new_list.append(stm.stem(word))
        else:
            for word in words:
                if(word in self.model.wv.vocab.keys()):
                    new_list.append(word)
        
        return list(set(new_list))


    def evaluate_topic(self, topic, metric=None):
        coher = []
        words = self.get_valid_words(topic)
        if(metric=="centroid"):
            centroid = self.calculate_centroid(words)
            for word in words:
                coher.append(1 - distance.cosine(self.model[word], centroid))
        else:
            combs = combination(words)
            for combin in combs:
                coher.append(self.pair_coherence(combin[0], combin[1], metric=metric))

        return np.mean(np.array(coher))

    def rank_topic(self, topic):
        words = self.get_valid_words(topic)
        centroid = self.calculate_centroid(words)
        
        coher = []
        for word in words:
            coher.append([word, 1 - distance.cosine(self.model[word], centroid)])
        
        if (self.stem):
            coher = [[self.rev_dic[x[0]],x[1]] for x in coher]
        coher.sort(key=lambda x:x[1], reverse=True)
        return coher

    def calculate_centroid(self, words):
        acum_vec = self.model[words[0]].copy()
        for word in words[1:]:
            acum_vec += self.model[word]
        centroid = acum_vec/len(words)
      
        return centroid
           
           
           
           
           
           