#!/bin/bash

# dependencies for a-bott
# I am using python 3.6.9, Ubuntu 18.04

# create a virtual environment for a-bott
# need to install virtualenvwrapper before this
source virtualenvwrapper.sh
mkvirtualenv abott_env -p python3

# activate virtual environment
workon abott_env

# install python modules
pip install numpy
pip install nltk
# may need to open a python shell and type
# import nltk
# nltk.download('punkt')
pip install tflearn
# for me, newer versions of tensorflow not working
pip install tensorflow==1.14
# to install pyaudio to ubuntu 18.04 in virtual environment:
sudo apt-get install portaudio19-dev python-pyaudio
pip install pyaudio
pip install SpeechRecognition
# for text to speech
pip install pyttsx3
sudo apt-get install espeak

# deactivate virtual environment
deactivate


