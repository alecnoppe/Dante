import gensim.downloader
from gensim.models import Word2Vec, word2vec
from gensim.test.utils import common_texts
from nltk.corpus import brown
import numpy as np
from src.kNN_model import KNN

class W2V_SentenceModel(KNN):
    def __init__(self):
        self.w2v = Word2Vec.load("brown_model.model")
    
    def similarity(self, v, w):
        v_included_words = [word for word in v.split(" ") if word in self.w2v.wv]
        w_included_words = [word for word in w.split(" ") if word in self.w2v.wv]
        return self.w2v.n_similarity(v_included_words, w_included_words)

class W2V_NameModel:
    def __init__(self):
        self.w2v = Word2Vec.load("brown_model.model")
        self.circles = {"Limbo":1, "Lust":2, "Gluttony":3, "Avarice":4, "Prodigality":4, "Wrath":5, 
            "Sullenness":5, "Heresy":6, "Violence":7, "Fraud":8, "Treachery":9}
        self.simple_circles = {"limbo":1, "lust":2, "hunger":3, "greed":4, "wrath":5, "heresy":6, 
            "violence":7, "fraud":8, "betrayal":9}
    
    def similarity(self, v):
        if v in self.w2v.wv:
            similarities = []
            for name in self.simple_circles.keys():   
                similarities.append([self.w2v.wv.similarity(v, name), self.simple_circles[name]])
            similarities = np.array(similarities)
            return int(similarities[similarities.argmax(axis=0)[0]][1])
        else:
            return -1
