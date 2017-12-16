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
