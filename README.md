"# RL_portf_optimisation

############## Data: 

Data is present in the 'RawData.csv' file.


############## Supervised Learning files: 

supervised.ipynb: Trains on data from 'RawData.csv'

Different sections of the notebook conatin functions on data formatting, model architecture, training and testing.

Call the function run_index_scaled(timestep, repeats, n_batch, n_epochs, n_neurons, train_up, train_low, test_up, test_low) to 

train the network on data starting from 'train_low' till 'train_up' and testing on data from 'test_low' till 'test_up'.

The output is of the form mean squared errors, predicted returns and expected returns



############## Q-Learning files:

Use the ICA-Preparation file to do ICA decomposition of the datasetm the result will be saved in the Ind_components.csv, ad a demixing .csv  file in the working directory, These files correspond to the decomposition in independent factors and to the demixing matrix.

The Q-learning file takes the Ind_components file as an imput performs the Q-update depending on the choosen model and the level of lambda if the Q-lambda model is choosen and performs the trading strategy. 
So the parameters are :
-the right Q-update function in the Q error function : dependent on us wanting to use Q-lambda or not 
-Lambda
-the discount factor : gamma
-the learning rate nu : nu 
-the error rate : error 
-the training date : train_size (corresponding to the length of an episode)
-the number of episodes: num_iterations
-The E greedy exploration : epsilon

The output is in the form of precision of the signal compared to the actual market signals, the expected cumulative returns using the trained policy, and the trained Q-table,



############## RNN Reinforcement Learning files:

####

RNN_learn.py : learns from 'RawData.csv'

Please modify the 'data_path' string below to the directory where 'RawData.csv' is stored

Egs: data_path = '/Users/AllData/'

Saves the model parameters in the location (data_path + folder_name + '/')

where, 'folder_name' is determined in run-time. Its determination can be modified in the code

####



####

RNN_test.py : tests the model's performance on test data

Please modify the 'data_path' string below to the directory where 'RawData.csv' is stored

Egs: data_path = '/Users/AllData/'

Note: 'folder_name' string below should be the name of the folder (inside the 'data_path' folder) in which the parameters were exported in the learning phase

####

"
