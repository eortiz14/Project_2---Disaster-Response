# Project_2 Disaster Response

## Table of Contents

    1.Requirements
    2.Project Motivation
    3.Instructions
    4.Licensing, Authors, and Acknowledgements

## 1.Requirements

Developed using Python 3.6 and the next python packages:

  * Pandas
  * Numpy
  * Scikit-learn
  * NLP from nltk
  * sqlalchemy
  * Pickle

## 2.Project Motivation

This project is a requirement of the Udacity Data Science Nanodegree. The main idea of the project is to create a disaster response pipeline using supervised machine learning model that helps categorize different messages recived during natural disasters. The process consist taking two csv files that have real messages during a natural disasters, merge them and clean the data for export it to a sql database. With the final database we have to build a NLP pipeline for classificate the messages in 36 categories. The last step is build a flask app that have data vizualization and the classification model.

## 3. Instructions

The repository contains:

* The data folder contains two csv files, process_data.py and a Jupyter Notebook that contains the similar code that process_data.py
* The models folder contains train_classifier.py, which builds and trains the model that will categorize messages
* The app folder contains run.py which is used to deploy the flask app

For deploy the flask app running make sure to be in the app directory use cd app in the terminal. Open a new terminal window and type in the command line: 'python run.py', then 
open another Terminal Window and type env|grep WORK, the result of this command put on a new web browser window, type in the following: https://SPACEID-3001.SPACEDOMAIN.

## 4. Licensing, Authors, Acknowledgements
This project was done by  Esteban Ortiz Gonzalez, an Industrial Engineer of the Pontificia Universidad Javeriana that currently work in Banco Davivienda.
Special acknowledgments to Udacity and Figure Eight for disaster data.
