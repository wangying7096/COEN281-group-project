# COEN281-group-project
There are a lot of algorithms developed for flight delay prediction, however,  for taxi-in time, it seems that no one has worked on it before. Taxi-in is the time duration elapsed between wheels-on and gate arrival at the destination airport. This information is actually very important and useful for the airport to prepare for the landing, wisely dispatch the airport runway for each flight.
Herein, our objective is to try using four different prediction models, including linear regression, polynomial regression, decision tree model with gradient boosting, and artificial neural network, to predict the taxi-in time using the history data and compare the results. 

# Excution instructions
## neuralNetwork
The dataset '2018.csv' should be located in the same folder as the source code. 
In the command line, entering "python neuralNetwork_all_airlines.py" will start executing the program.
## Gradient Boosting Regressor
The dataset should be located in the same folder as the source code, and the name sould be 2018.csv. 
By typing the command line "Python GradientBoostingRegressor.py" will start train the model and give MSE of the test result.
## Linear/Polynomial Regression
The dataset should be located in the same folder as the source code, and the name sould be 2018.csv. 
By typing the command line "python preprocessing.py" will start preprocessing the data
After pure_data.pkl and label.pkl is generated,type in "python model.py"
The program will print out mse and cross validation result
