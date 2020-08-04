import torch as t
from sklearn.metrics import f1_score
from tqdm.autonotebook import tqdm
import torch.nn as nn


class Trainer:

    def __init__(self,
                 model,  # Model to be trained.
                 crit,  # Loss function
                 optim=None,  # Optimizer
                 train_dl=None,  # Training data set
                 val_test_dl=None,  # Validation (or test) data set
                 cuda=True,  # Whether to use the GPU
                 early_stopping_patience=-1):  # The patience for early stopping
        self._model = model
        self._crit = crit
        self._optim = optim
        self._train_dl = train_dl
        self._val_test_dl = val_test_dl
        self._cuda = cuda

        self._early_stopping_patience = early_stopping_patience

        if cuda:
            self._model = model.cuda()
            self._crit = crit.cuda()

    def save_checkpoint(self, epoch):
        t.save({'state_dict': self._model.state_dict()}, 'checkpoints/checkpoint_{:03d}.ckp'.format(epoch))

    def restore_checkpoint(self, epoch_n):
        ckp = t.load('checkpoints/checkpoint_{:03d}.ckp'.format(epoch_n), 'cuda' if self._cuda else None)
        self._model.load_state_dict(ckp['state_dict'])

    def save_onnx(self, fn):
        m = self._model.cpu()
        m.eval()
        x = t.randn(1, 3, 300, 300, requires_grad=True)
        y = self._model(x)
        t.onnx.export(m,  # model being run
                      x,  # model input (or a tuple for multiple inputs)
                      fn,  # where to save the model (can be a file or file-like object)
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=10,  # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names=['input'],  # the model's input names
                      output_names=['output'],  # the model's output names
                      dynamic_axes={'input': {0: 'batch_size'},  # variable lenght axes
                                    'output': {0: 'batch_size'}})

    def train_step(self, inputs, labels):
        # self._model.zero_grad()
        self._optim.zero_grad()  # -reset the gradients
        out = self._model.forward(inputs.cuda())  # -propagate through the network
        loss = self._crit(out, labels.float().cuda())  # -calculate the loss
        loss.backward()  # -compute gradient by backward propagation
        self._optim.step()  # -update weights
        return loss  # -return the loss

    def val_test_step(self, input, labels):

        # predict
        out = self._model.forward(input.cuda())  # -propagate through the network
        loss = self._crit(out, labels.float().cuda())
        # propagate through the network and calculate the loss and predictions
        # return the loss and the predictions
        # TODO
        return loss, out

    def train_epoch(self):
        # self._train_dl.mode = 'train'  # set training mode
        loss = []
        for i, data in enumerate(self._train_dl, 0):  # iterate through the training set
            inputs, labels = data
            if self._cuda:
                self._model.cuda()  # transfer the batch to "cuda()" -> the gpu if a gpu is given
            loss.append(self.train_step(inputs, labels))  # perform a training step

        avg = t.mean(t.tensor(loss))

        # calculate the average loss for the epoch and return it
        return avg

    def val_test(self):
        loss = []
        labels_list = []
        with t.no_grad():
            for data in self._val_test_dl:
                images, labels = data
                labels_list.append(labels)
                if self._cuda:
                    self._model.cuda()
                loss.append(self.val_test_step(images, labels)[0])
        # print(labels)
        # print(loss)
        # print(f1_score(labels_list, loss, average='macro'))
        avg = t.mean(t.tensor(loss))
        # set eval mode
        # disable gradient computation
        # iterate through the validation set
        # transfer the batch to the gpu if given
        # perform a validation step
        # save the predictions and the labels for each batch
        # calculate the average loss and average metrics of your choice. You might want to calculate these metrics in designated functions
        # return the loss and print the calculated metrics
        return avg

    def fit(self, epochs=-1):
        assert self._early_stopping_patience > 0 or epochs > 0
        # create a list for the train and validation losses, and create a counter for the epoch
        train_loss = []
        val_loss = []
        counter = 0
        break_flag = False
        new_count = 0
        while True:
            if counter >= epochs:
                print('out of epochs')
                break

            train_loss.append(self.train_epoch())
            val_loss.append(self.val_test())
            if counter == 0:
                lowest = val_loss[counter]
            else:
                if lowest > val_loss[counter]:
                    print(lowest,'Higher than', val_loss[counter])
                    new_count += 1
                else:
                    print('lower')
                    lowest = val_loss[counter]
                    new_count = 0
            if new_count == self._early_stopping_patience:
                print('verbessert sich nicht')
                return train_loss, val_loss
            print(train_loss[counter], val_loss[counter])
            counter = counter + 1
            # stop by epoch number
            # train for a epoch and then calculate the loss and metrics on the validation set
            # append the losses to the respective lists
            # use the save_checkpoint function to save the model (can be restricted to epochs with improvement)
            # check whether early stopping should be performed using the early stopping criterion and stop if so
            # return the losses for both training and validation
        return train_loss, val_loss
