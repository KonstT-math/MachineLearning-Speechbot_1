

# project: chatbot with speech recognition

# note: when ran in python 3.6.9, several warning issues occur.

# modules to download: 
# numpy,
# nltk, tflearn, tensorflow 
# (note: tensorflow 2.* does not work properly
# - try pip install tensorflow==1.14 or older)

# 1) chatbot with machine learning
# 2) enrich with speech recognition via google api

import nltk
# also, get into a python shell and type:
# >>> import nltk
# >>> nltk.download('punkt')
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import numpy
import tflearn
import tensorflow
import random
import json
import pickle

# opens the intents.json file with the patterns and responces
# and saves it to data
with open("intents.json") as file:
	data = json.load(file)

# will keep the tokenized words in words
words = []
# will keep "tag"s in labels
labels = []
docs_x = []
docs_y = []

# runs through "intents"
for intent in data["intents"]:
	# runs through "patterns" in "intents"
	for pattern in intent["patterns"]:
		# tokenizes the words in "patterns"
		wrds = nltk.word_tokenize(pattern)
		words.extend(wrds)
		# stores the tokenized word in docs_x
		docs_x.append(wrds)
		docs_y.append(intent["tag"])

	if intent["tag"] not in labels:
		labels.append(intent["tag"])

# normalize and sort words in the following way:
words = [stemmer.stem(w.lower()) for w in words if w != "?"]
words = sorted(list(set(words)))
# sort labels
labels = sorted(labels)

# training and testing output:
# up to now, we have strings
# neural networks work with numbers.. this is our next step..

# preprocessing..

# encode words: for input/ouput to neural network
# [0,1,0,0,0,1,1,0,0,0,1,1,0] - if exists (1) or not (0)
# example for "greeting":
# [1,1,0,0]
# "hi", "hey", "sell", "help"

training = []
output = []

out_empty = [0 for _ in range(len(labels))]
for x, doc in enumerate(docs_x):
	bag = []
	# stem words in doc_x
	wrds = [stemmer.stem(w) for w in doc]

	for w in words:
		if w in wrds:
			bag.append(1)
		else:
			bag.append(0)

	output_row = out_empty[:]
	output_row[labels.index(docs_y[x])] = 1

	training.append(bag)
	output.append(output_row)

training = numpy.array(training)
output = numpy.array(output)

# write bytes, as seen below, in data.pickle file
with open("data.pickle", "wb") as f:
		pickle.dump((words, labels, training, output),f)

# build model with tflearn:
# - input layer (of length len(training[0])) 
# - n hidden layers (of length m each)
# - output layer (of length len(output[0]))
# with "softmax" activation function, that gives probability to each output neuron

# with more neuron hidden layers (and more epochs) gets probably better;
# but for this app, 1 or 2 is ok..

tensorflow.reset_default_graph()
# input layer:
net = tflearn.input_data(shape=[None, len(training[0])]) 
# 2 fully connected hidden layer with 15 nuerons:
net = tflearn.fully_connected(net,15)
net = tflearn.fully_connected(net,15)
# fully connected hidden layers with a number of neurons:
#net = tflearn.fully_connected(net,12) 
# output layer (softmax: to get probabilities for each output):
net = tflearn.fully_connected(net, len(output[0]), activation="softmax") 
net = tflearn.regression(net)
# to train model:
# DNN is a type of neural network
model = tflearn.DNN(net)

# to fit our model:
# n_epoch=500 by trial and error - try other values
model.fit(training, output, n_epoch=500, batch_size=8, show_metric=True)
# save model as model.tflearn
model.save("model.tflearn")
