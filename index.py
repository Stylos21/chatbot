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
output = []
patterns = []
with open('index.json') as f:
    f = json.load(f)
    patterns.append([f["intents"][i]["patterns"] for i in range(len(f["intents"]))])
    patterns = patterns[0]
    output.append([f["intents"][i]["responses"] for i in range(len(f["intents"]))])
    output = output[0]
    ys.append([f['intents'][i]["id"] for i in range(len(f["intents"]))])
    ys = ys[0]

    #print(len(f['intents']))
    for i in range(len(f["intents"])):
        for intent in f["intents"]:
            print(intent['id'])
            # print(f["intents"].index(intent['patterns'][i] for i in range(len(intent["patterns"]))))
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

##print(xs, ys)
xs = list(set(xs))
ys = list(set(ys))

# print(ys)
xs_train = [0 for _ in range(len(xs))]

ys_train = [[0 for _ in range(len(ys))] for _ in range(len(ys))]

#print(ys_train)
print(patterns)

for i, c in enumerate(sorted(xs)):
    for word in range(len(xs)):
        #print(c, word)
        if xs[word] in c:
            
            xs_train[word] = 1

        else:
            xs_train[word] = 0


    #print(xs_train)

for i in range(len(patterns)):
    for ii in range(len(patterns[i]) - 1):
        appendable = [0 for _ in range(len(patterns))]
        appendable[patterns.index(patterns[i])] = 1
        print(appendable)
        ys_train.append(appendable)
ys_train = ys_train[6:]
print(ys_train)


# for i,c in enumerate(ys):
#     for word in range(len(ys)):
#         if ys[word] in c:
#             ys_train[word] = i / 6            
#print(len(xs_train), len(ys_train))
xs_train = np.array([xs_train])
ys_train = np.array(ys_train)
# ys_train.reshape((-1, 7))
#print(ys_train)
#print(xs_train.shape, ys_train.shape)
tf.reset_default_graph()
net = tflearn.input_data(shape=[None, len(xs_train[0])])
net = tflearn.fully_connected(net, 64, activation="relu")
net = tflearn.fully_connected(net, 64, activation="relu")
net = tflearn.fully_connected(net, 6, activation="softmax")
net = tflearn.regression(net)
model = tflearn.DNN(net)
# model.fit(xs_train, ys_train, n_epoch=5000)
# model.save('index.model')
model.load('index.model')
def predict(words, vocab):
    words = word_tokenize(words)
    words = [word.lower() for word in words]
    arr = [0 for _ in range(len(vocab))]
    for i, c in enumerate(vocab):
        for ii in range(len(words)):
            if words[ii] in c:
                arr[ii] = 1

    return np.array([arr])

prediction = model.predict(predict("how old are you", xs))
res = np.argmax([prediction])
print(ys[res], res, ys)

if ys[res] == "greeting":
    responses = output[0]
    print(random.choice(responses))

elif ys[res] == "stats":
    responses = output[1]
    print(random.choice(responses))

elif ys[res] == "age":
    responses = output[2]
    print(random.choice(responses))

elif ys[res] == "howareyou":
    responses = output[3]
    print(random.choice(responses))

elif ys[res] == "about":
    responses = output[4]
    print(random.choice(responses))
elif ys[res] == "hobbies":
    responses = output[5]
    print(random.choice(responses))






print(patterns)
