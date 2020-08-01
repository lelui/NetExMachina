import torch as t
from trainer import Trainer
from matplotlib import pyplot as plt
import numpy as np
import model
import pandas as pd
from sklearn.model_selection import train_test_split
from data import ChallengeDataset
import torchvision


# load the data from the csv file and perform a train-test-split
# this can be accomplished using the already imported pandas and sklearn.model_selection modules
csvthing = pd.read_csv('data.csv')
train ,test = train_test_split(csvthing, train_size=0.75, test_size=0.25)
data_train = ChallengeDataset(train, 'train')
data_test = ChallengeDataset(test, 'val')

# set up data loading for the training and validation set each using t.utils.data.DataLoader and ChallengeDataset objects
test_loader = t.utils.data.DataLoader(data_test, batch_size=15)
train_loader = t.utils.data.DataLoader(data_train, batch_size=5)
# create an instance of our ResNet model
netexmachina = model.ResNet()

# set up a suitable loss criterion (you can find a pre-implemented loss functions in t.nn)
# set up the optimizer (see t.optim)
# create an object of type Trainer and set its early stopping criterion
loss_crit = crit = t.nn.BCELoss()
optim = t.optim.Adam(netexmachina.parameters(), lr=0.01)
patience = 5
AshKetchum = Trainer(netexmachina, loss_crit, optim, train_dl=train_loader, val_test_dl=test_loader, cuda=True, early_stopping_patience=5)
# go, go, go... call fit on trainer
res = AshKetchum.fit(100)#TODO

# plot the results
plt.plot(np.arange(len(res[0])), res[0], label='train loss')
plt.plot(np.arange(len(res[1])), res[1], label='val loss')
plt.yscale('log')
plt.legend()
plt.savefig('losses.png')