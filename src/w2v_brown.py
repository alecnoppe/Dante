from gensim.models import Word2Vec
from nltk.corpus import brown

model = Word2Vec(brown.sents(), vector_size=100, window=5, min_count=1, workers=4 )
model.save("brown_model.model")