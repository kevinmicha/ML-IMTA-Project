# Import PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data_utils
from torch.utils.data.sampler import SubsetRandomSampler

from clean_normalize import *
from ml_functions import *

# Define network architecture
class Net1(nn.Module):
    def __init__(self,nb_features):
        super(Net1,self).__init__()
        self.fc1 = nn.Linear(nb_features, 1)    
    def forward(self,x):
        out = self.fc1(x)
        out = torch.sigmoid(out)
        return out
        
def create_torch_dataset(X_train, X_test, y_train, y_test, batch_size=20):
    '''
    Making a torch-type Dataset

    INPUT
    X_train: features for training
    X_test: features for testing
    y_train: targets for training
    y_test: targets for testing
    batch_size: number of samples processed before the model is updated

    OUTPUT
    train_loader: pytorch data-loader for training
    test_loader: pytorch data-loader for testing

    AUTHOR
    Ezequiel Centofanti
    '''
    X_train_t = torch.Tensor( np.array(X_train) ) 
    y_train_t = torch.Tensor( np.array(pd.DataFrame(y_train)) )
    y_train_t = y_train_t.type(torch.LongTensor)
    y_train_t = y_train_t.to(torch.float32)
    data_train = data_utils.TensorDataset(X_train_t, y_train_t)
    train_loader = torch.utils.data.DataLoader(data_train, batch_size = batch_size)

    X_test_t = torch.Tensor( np.array(X_test) ) 
    y_test_t = torch.Tensor( np.array(pd.DataFrame(y_test)) )
    y_test_t = y_test_t.type(torch.LongTensor)
    y_test_t = y_test_t.to(torch.float32)
    data_test = data_utils.TensorDataset(X_test_t, y_test_t)
    test_loader = torch.utils.data.DataLoader(data_test, batch_size = batch_size)

    return train_loader, test_loader

def training(n_epochs, train_loader, model, criterion, optimizer):
    '''
    Training a torch model

    INPUT
    n_epochs: number of training loops over all the dataset
    train_loader: pytorch data-loader for training
    model: torch model to train
    criterion: loss criterion
    optimizer: optimizer method

    OUTPUT
    train_losses: array of losses at each epoch

    AUTHOR
    Ezequiel Centofanti
    '''
    train_losses = []

    for epoch in range(n_epochs):
        train_loss = 0 # monitor losses
        # train the model
        model.train() # prep model for training
        for data, label in train_loader:
            optimizer.zero_grad() # clear the gradients of all optimized variables
            output = model(data) # forward pass: compute predicted outputs by passing inputs to the model
            loss = criterion(output, label) # calculate the loss
            loss.backward() # backward pass: compute gradient of the loss with respect to model parameters
            optimizer.step() # perform a single optimization step (parameter update)
            train_loss += loss.item() * data.size(0) # update running training loss
        
        train_loss /= len(train_loader.sampler)
        train_losses.append(train_loss)

        #print('epoch: {} \ttraining Loss: {:.6f}'.format(epoch+1, train_loss))

    return train_losses


def evaluation(model, test_loader, criterion):
    '''
    Evaluating a torch model and printing the accuracy over the test-set

    INPUT
    model: torch model to train
    test_loader: pytorch data-loader for testing
    criterion: loss criterion

    OUTPUT
    y_predicted: predicted labels for the testing set

    AUTHOR
    Ezequiel Centofanti
    '''
    # initialize values to monitor test accuracy
    pred_correct = 0
    pred_total = 0
    
    y_predicted = []

    model.eval() # prep model for evaluation
    for data, label in test_loader:
        with torch.no_grad():
            output = model(data) # forward pass: compute predicted outputs by passing inputs to the model
        pred = output > 0.5
        y_predicted.extend([int(np.array(pred)[i][0]) for i in range(len(pred))])
        correct = np.squeeze(pred) == (np.squeeze(label)==1) # compare predictions to true label
        # calculate test accuracy for each batch
        for i in range(len(label)):
            pred_correct += correct[i].item()
            pred_total += 1

    # Calculate and print avg test accuracy
    print('test accuracy: %2d%% (%2d/%2d)' % ( 100 * pred_correct / pred_total, pred_correct, pred_total))
    return y_predicted