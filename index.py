import tensorflow as tf 
import nltk 
from nltk.tokenize import word_tokenize
from nltk.stem import LancasterStemmer
stemmer = LancasterStemmer()
import tflearn 
import random  
import json 
import numpy as np   
xs = []
ys = []
with open('index.json') as f:
    
    f = json.load(f)
    for intent in f["intents"]:
        ys.append(intent['id'].lower())
        # for tag in intent["id"]:
        #     print(tag)
        for pattern in intent["patterns"]:
            pattern = word_tokenize(pattern)
            for i in range(len(pattern)):
                xs.append(pattern[i].lower())


        for response in intent["responses"]:
            response = word_tokenize(response)
            for i in range(len(response)):
                xs.append(response[i].lower())

#print(xs, ys)
xs = list(set(xs))
ys = list(set(ys))

# print(ys)
xs_train = [0 for _ in range(len(xs))]
ys_train = [0 for _ in range(len(ys))]
for i, c in enumerate(sorted(xs)):
    for word in range(len(xs)):
        if xs[word] in c:
            xs_train[word] = 1

        else:
            xs_train[word] = 0

for i,c in enumerate(ys):
    for word in range(len(ys)):
        if ys[word] in c:
            ys_train[word] = i / 7            
print(len(xs_train), len(ys_train))
xs_train = np.array([xs_train])
ys_train = np.array([ys_train])
# ys_train.reshape((-1, 7))

print(xs_train.shape, ys_train.shape)
tf.reset_default_graph()
net = tflearn.input_data(shape=[None, len(xs_train[0])])
net = tflearn.fully_connected(net, 64, activation="softmax")
net = tflearn.fully_connected(net, 64, activation="softmax")
net = tflearn.fully_connected(net, 7, activation="sigmoid")
net = tflearn.regression(net)
model = tflearn.DNN(net)
model.fit(xs_train, ys_train, n_epoch=10000)

# model.load('index.model')
def predict(words, vocab):
    words = word_tokenize(words)
    words = [word.lower() for word in words]
    arr = [0 for _ in range(len(vocab))]
    for i, c in enumerate(vocab):
        for ii in range(len(words)):
            if words[ii] in c:
                arr[ii] = 1

    return np.array([arr])

prediction = model.predict(predict("la", xs))
res = np.argmax([prediction])
with open('index.json') as f:
    f = json.load(f)
    responses = f["intents"][res]['responses']

    print(random.choice(responses))


print(f"{ys[np.argmax([prediction])], prediction[0][res]}")



