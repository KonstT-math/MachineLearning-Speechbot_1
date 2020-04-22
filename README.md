# MachineLearning-Speechbot
A Machine Learning implemented Speech Bot, which listens via Google API and responds via espeak and pyttsx3 library.

Implemented in Python.

1) The install.sh is a shell script containing the main Python modules that are needed for the setup. It creates a virtual environment named abott_env. This is being done via virtualenvwrapper.

2) The json file intents.json can be filled with patterns and responses in order to be used for chat

3) Run the training.py in order to train the model. I have used a DNN with 2 fully connected hidden layers with 15 neurons each. The architecture can be adjusted by trial and error, depending on the contents of the json file. 

4) The main.py contains the main algorithm which consists of two functions; one for chatting via keyboard, and one for using the google API so that the bot listens for spoken sentenses, and responds via pyttsx3 library and espeak (for linux).

5) The acc_stats.csv is supposed to be filled with data in order to run statistics on various models used, comparing the resulting accuracy.
