import numpy as np
from io import StringIO
from src.random_model import RandomModel
from src.w2v_model import W2V_SentenceModel, W2V_NameModel
from src.evaluate_model import evaluate

train = np.genfromtxt("data/train.csv", delimiter=",", skip_header=1, names=True, dtype=None, encoding='UTF-8')
test = np.genfromtxt("data/test.csv", delimiter=",", skip_header=1, names=True, dtype=None, encoding='UTF-8')

X_train = np.array([train[x][0] for x in range(0,len(train))])
y_train = np.array([train[x][1] for x in range(0,len(train))])

X_test = np.array([test[x][0] for x in range(0,len(test))])
y_test = np.array([test[x][1] for x in range(0,len(test))])

random_model = RandomModel()
random_model.fit(X_train, y_train)
y_hat = random_model.predict(X_test)

print(np.mean(y_hat==y_test))

KNN_Name = W2V_NameModel()
word = "terror"
print(f'You go to Circle {KNN_Name.similarity(word)} of hell')