# Among Us Impostor Detection

Work in progress for a framework and ML algorithms for detecting impostors in the game Among Us

The main idea is to extract game information (chat messages, crew movement) and give it to classifiers such that they can determine if someone is an imposter or not.


## Features

- Extracting text messages through (Tesseract) OCR and putting them in JSON format for analysis (used for text classification in our project).
- Detecting which color pressed a button for a meeting or reported the body
- Extract imposter data if confirmed ejects are on (for generating training data for supervised learning)
- Detecting which crewmates are on the screen
- Detecting tasks, and if crewmates are doing them (and potentially if they are faking them)

## Goals

For the natural languae processing:
Have text classifiers which can determine if a chat message is something most likely said by an impostor or crewmate
For this, we use several classifiers (Naive Bayes , SVM )

For the machine vision:
Use a CNN (convolutional neural network) for classification


## Framework in action

Some pictures showing what the framework can do:

Text detection:

<img width="60%" src ="https://i.imgur.com/TeyntnB.png"/><br>

Applying OCR (using Tesseract) to the detected text and turning the chat into a JSON file:

<img width="30%" src ="https://i.imgur.com/bCNqVxb.png"/><br>


## contact

My email: letabot (a) gmx (.) com 

This project is part of Serpentine: https://serpentineai.nl/contact/
