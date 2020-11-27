# Disaster Response Pipeline Project

# Table of Contents

[1. Project Motivation](#Project-Motivation)

[2. Installation](#Installation)

[3. File Description](#File-Description)

[4. Instructions](#Instructions)

# Project Motivation
For this project, we used the disaster data from Figure Eight to build a model to classify disaster messages. A pipeline and a web app are built such that future new messages can be processed smoothly and that results can be visualized easily.  

# Installation
Below are the libraries require for this project: 
SQLAlchemy==1.2.1 
numpy==1.14.0 
Flask==0.12.2 
nltk==3.2.5 
pandas==0.22.0 
plotly==4.13.0 
scikit_learn==0.23.2  

# File Description
* data folder contains:
  * disaster_categories.csv: disaster categories csv file
  * disaster_messages.csv: disaster messages csv file
  * process_data.py: script to transform disaster_categories.csv and disaster_messages to DisasterResponse.db. 
  * DisasterResponse.db: A database file that is a merge of categories and messages. This is outputted by prcoess_data.py.
* models folder contains:
  * classifier.pkl: pickle file of classifier 
  * train_classifies.py: script in building the model
* app folder contains:
  * run.py: run the pipeline as Flask application
* requirements.txt: libraries required for the project

# Instructions
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
