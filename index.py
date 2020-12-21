import json
import tensorflow as tf	
from tensorflow import keras 
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
train_labels = np.array(train_labels)

model = tf.keras.Sequential()	

model.add(tf.keras.layers.InputLayer(input_shape=(len(new_xs[0]))))	
model.add(tf.keras.layers.Dense(64, activation="relu"))	
model.add(tf.keras.layers.Dense(64, activation="relu"))
model.add(tf.keras.layers.Dense(len(labels), activation="softmax"))
	
def train():
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])	  
    model.fit(new_xs, train_labels, epochs=1000, batch_size=256)	  
    model.save('model.h5')

train()



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
