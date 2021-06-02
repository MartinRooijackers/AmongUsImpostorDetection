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




## setup:

For this project we used Pycharm for coding, but any python IDE will work as long as you can pip the libraries needed.

You will need the .pb model for the EAST text detection:

https://www.dropbox.com/s/r2ingd0l3zt8hxs/frozen_east_text_detection.tar.gz?dl=1

You will also need tesseract.
The (compiled) downloads for tesseract (windows/linux) can be found here:

https://digi.bib.uni-mannheim.de/tesseract/

5.0 was used, but any version 4 version should work as well (since no new 5.0 features are used)

### setting variables

After those two files are downloaded (and installed in the case of tessearct), you will need to set 3 variables in the framework.python


**Tesseract_location** :  path to the install location of tesseract. Make sure to include the / at the end as well

**model_detector** : Path to a .pb file contains trained detector network:


**video_location** : folder which contains all the videos you want to analyze. Make sure to include the / at the end as well

Once all those are set propperly, put the videos in your video location folder. When you run the framework.py it should now be analyzing the videos there.


## Framework in action

Some pictures showing what the framework can do:

Text detection:

<img width="60%" src ="https://i.imgur.com/TeyntnB.png"/><br>

Applying OCR (using Tesseract) to the detected text and turning the chat into a JSON file:

<img width="30%" src ="https://i.imgur.com/bCNqVxb.png"/><br>


## contact

My email: letabot (a) gmx (.) com 

This project is part of Serpentine: https://serpentineai.nl/contact/
