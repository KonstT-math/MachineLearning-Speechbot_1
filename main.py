
# project: chatbot with speech recognition

# note: when ran in python 3.6.9, several warning issues occur.

# modules to download: 
# numpy,
# nltk, tflearn, tensorflow 
# (note: tensorflow 2.* does not work properly
# - try pip install tensorflow==1.14 or older)
# for speech to text
# pyaudio, SpeechRecognition
# for text to speech
# pyttsx3

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

import speech_recognition as sr
import pyttsx3

# open and read intents.json and data.pickle files:

with open("intents.json") as file:
	data = json.load(file)

with open("data.pickle", "rb") as f:
	words, labels, training, output = pickle.load(f)

# describe and load the model, and its architecture (as in the main_abott.py file)
net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net,15)
net = tflearn.fully_connected(net,15)
net = tflearn.fully_connected(net, len(output[0]), activation='softmax')
net = tflearn.regression(net)
model = tflearn.DNN(net, tensorboard_dir='tflearn_logs')

model.load("model.tflearn")

def bag_of_words(s, words):
	bag = [0 for _ in range(len(words))]
	
	s_words = nltk.word_tokenize(s)
	s_words = [stemmer.stem(word.lower()) for word in s_words]
	
	for se in s_words:
		for i, w in enumerate(words):
			if w == se:
				bag[i] = 1
	
	return numpy.array(bag)

# function used for "speak" mode:
# listens speech via google API, and uses the trained model to respond
def chat_speakmode():
	# for listening:
	r=sr.Recognizer()
	mic=sr.Microphone()
	# for speaking:
	engine = pyttsx3.init()
	rate = engine.getProperty('rate')
	engine.setProperty('rate', rate-90)
	#voices = engine.getProperty('voices')
	#engine.setProperty("voice", voices[50].id)
	# or
	#engine.setProperty('voice', 'greek+f3')
	#engine.setProperty('voice', 'english+f3')

	print("Start talking with the bot! (say \"shut down\" to stop)")
	while True:
		while True:		
			try:
				print("You: ")
				with mic as source:
					r.adjust_for_ambient_noise(source,duration=0.5)
					audio=r.listen(source)
				inp = r.recognize_google(audio)
				print(inp)
				break
			except sr.UnknownValueError:
				print("Abott: Could not hear ya, say again")
				engine.say('Abott could not hear ya, say again')
				engine.runAndWait()
		
		if inp.lower() == "shut down":
			break

		results = model.predict([bag_of_words(inp, words)])
		# to print the probabilities of each output neuron
		# print(results)
		# to return the index of the greatest probability
		results_index = numpy.argmax(results)
		# put the index to labels
		tag = labels[results_index]
		# prints the tag it thinks we should be in:
		# print(tag)
		for tg in data["intents"]:
			if tg["tag"] == tag:
				responses = tg["responses"]
		a_response = random.choice(responses)
		print("Abott: {}".format(a_response))
		engine.say(a_response)
		engine.runAndWait()

# function used for "type" mode:
# reads the typed text, and uses the trained model to respond
def chat_typemode():
	print("Start chatting with the bot! (type shut down to stop)")
	while True:
		inp = input("You: ")
		if inp.lower() == "shut down":
			break

		results = model.predict([bag_of_words(inp, words)])
		# to print the probabilities of each output neuron
		# print(results)
		# to return the index of the greatest probability
		results_index = numpy.argmax(results)
		# put the index to labels
		tag = labels[results_index]

		for tg in data["intents"]:
			if tg["tag"] == tag:
				responses = tg["responses"]
		print("Abott: {}".format(random.choice(responses)))

flag = 0
flag = input('type or speak mode?')
if flag == 'speak':
	chat_speakmode()
else:
	chat_typemode()
