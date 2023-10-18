# LSTM
Stock Price Prediction using LSTM
This repository contains a Jupyter notebook that implements a Long Short-Term Memory (LSTM) model to predict the stock price of a given security. The notebook utilizes Python, Keras, and various APIs to fetch historical stock data, train the LSTM model, and make predictions.

Prerequisites
To run the notebook, you'll need to have the following installed:

Python 3
TensorFlow
Keras
Pandas
matplotlib
Usage
Clone this repository.
Open the LSTM_NEURAL.ipynb notebook in Jupyter.
Set the desired stock ticker symbol in the ASSET variable.
Run the notebook cells to train the LSTM model and make predictions.
APIs Used
The notebook utilizes the following APIs to fetch historical stock data:

Yahoo Finance API
Model Architecture
The LSTM model implemented in the notebook consists of an LSTM layer followed by a Dense layer for making predictions. The model is trained using historical stock data, capturing long-term dependencies and patterns in the stock price movements.

Disclaimer
This notebook is intended for educational and research purposes only. Stock prices are inherently unpredictable, and the predictions made by the model should not be considered financial advice
