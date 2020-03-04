import json
import tflearn
from random import randrange
import numpy as np
xs = []
ys = []
labels = []
vocab = []
new_xs = []

with open('index.json') as f:
    f = json.load(f)
    for intent in f['intents']:

        for _ in range(len(intent['patterns'])):
            ys.append(intent['id'])
            if intent['id'] not in labels:
                labels.append(intent['id'])
        for i in intent['patterns']:
            split = i.split(' ')
            xs.append(split)

            for word in split:
                if word not in vocab:
                    vocab.append(word)
train_labels = []
for label in ys:
    l = [0 for _ in range(len(labels))]
    l[labels.index(label)] = 1
    train_labels.append(l)
for x in xs:
    a = [0 for _ in range(len(vocab))]
    for word in x:
        if word in vocab:
            a[vocab.index(word)] = 1
    new_xs.append(a)
new_xs = np.array(new_xs)
net = tflearn.input_data(shape=[None, len(new_xs[0])])
net = tflearn.fully_connected(net, 64, activation="relu")
net = tflearn.fully_connected(net, 64, activation="relu")
net = tflearn.fully_connected(net, len(labels), activation="softmax")
net = tflearn.regression(net)
model = tflearn.DNN(net)
model.fit(new_xs, train_labels, n_epoch=1000, show_metric=True)

def chat():
    while True:
        text = input("Hello! What do you want to say? > ")
        a = text.split(' ')
        array = [0 for _ in range(len(vocab))]
        for word in a:
            if word in vocab:
                array[vocab.index(word)] = 1
        # array = np.reshape(array, ())
        pred = model.predict(np.array([array]))
        with open('index.json') as f:
            f = json.load(f)
            i = np.argmax(pred)
            r = f['intents'][i]['responses'][randrange(0, len(f['intents'][i]['responses']))]
            print(r)

chat()
