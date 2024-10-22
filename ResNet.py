import torch
import torch.nn.functional as F
import numpy as np
from torch import optim
from torchvision import datasets, transforms, models
import torch.nn as nn
from torch.nn.modules.batchnorm import _BatchNorm
import pandas as pd

# Resnet parameters
optimizer = 'sgd'
lr = 0.01
rho = 0.1
nEpoch = 100


def disable_running_stats(model):
    def _disable(module):
        if isinstance(module, _BatchNorm):
            module.backup_momentum = module.momentum
            module.momentum = 0

    model.apply(_disable)

def enable_running_stats(model):
    def _enable(module):
        if isinstance(module, _BatchNorm) and hasattr(module, "backup_momentum"):
            module.momentum = module.backup_momentum

    model.apply(_enable)

import os
import uuid

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data

class ResNetTrainer:
    def __init__(self, device, logger):
        self.device = device
        self.logger = logger
        self.tmp_dir = 'tmp'
        if not os.path.exists(self.tmp_dir):
            os.makedirs(self.tmp_dir)

    def fit(self, model, X_train, y_train, epochs=500, batch_size=128, eval_batch_size=128):
        file_path = os.path.join(self.tmp_dir, str(uuid.uuid4()))

        train_dataset = Dataset(X_train, y_train)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
        optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=4e-3)

        best_acc = 0.0

        # Training
        for epoch in range(epochs):
            model.train()
            for i, inputs in enumerate(train_loader, 0):
                X, y = inputs
                X, y = X.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                outputs = model(X)
                loss = F.nll_loss(outputs, y)
                loss.backward()
                optimizer.step()

            model.eval()
            acc = compute_accuracy(model, train_loader, self.device)
            if acc >= best_acc:
                best_acc = acc
                torch.save(model.state_dict(), file_path)
            self.logger.log('--> Epoch {}: loss {:5.4f}; accuracy: {:5.4f}; best accuracy: {:5.4f}'.format(epoch, loss.item(), acc, best_acc))

        # Load the best model
        model.load_state_dict(torch.load(file_path))
        model.eval()
        os.remove(file_path)

        return model

    def test(self, model, X_test, y_test, batch_size=128):
        test_dataset = Dataset(X_test, y_test)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=1)
        acc = compute_accuracy(model, test_loader, self.device)
        return acc

def compute_accuracy(model, loader, device):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in loader:
            X, labels = data
            X, labels = X.to(device), labels.to(device)
            outputs = model(X)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    acc = correct / total
    return acc

class ResNet(nn.Module):

    def __init__(self, input_size, nb_classes):
        super(ResNet, self).__init__()
        n_feature_maps = 64

        self.block_1 = ResNetBlock(input_size, n_feature_maps)
        self.block_2 = ResNetBlock(n_feature_maps, n_feature_maps)
        self.block_3 = ResNetBlock(n_feature_maps, n_feature_maps)
        self.linear = nn.Linear(n_feature_maps, nb_classes)

    def forward(self, x):
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)
        x = F.avg_pool1d(x, x.shape[-1]).view(x.shape[0],-1)
        x2 = self.linear(x)
        y = F.log_softmax(x2, dim=1)

        return y,x

class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResNetBlock, self).__init__()
        self.expand = True #if in_channels < out_channels else False

        self.conv_x = nn.Conv1d(in_channels, out_channels, 7, padding=3)
        self.bn_x = nn.BatchNorm1d(out_channels)
        self.conv_y = nn.Conv1d(out_channels, out_channels, 5, padding=2)
        self.bn_y = nn.BatchNorm1d(out_channels)
        self.conv_z = nn.Conv1d(out_channels, out_channels, 3, padding=1)
        self.bn_z = nn.BatchNorm1d(out_channels)

        if self.expand:
            self.shortcut_y = nn.Conv1d(in_channels, out_channels, 1)
        self.bn_shortcut_y = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        out = F.relu(self.bn_x(self.conv_x(x)))
        out = F.relu(self.bn_y(self.conv_y(out)))
        out = self.bn_z(self.conv_z(out))

        if self.expand:
            x = self.shortcut_y(x)
        x = self.bn_shortcut_y(x)
        out += x
        out = F.relu(out)

        return out

class Dataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)

    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def __len__(self):
        return len(self.X)

import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader

class Dataset(Dataset):
  def __init__(self, X_train, y_train):
    # need to convert float64 to float32 else
    # will get the following error
    # RuntimeError: expected scalar type Double but found Float
    self.X = torch.from_numpy(X_train.astype(np.float32))
    # need to convert float64 to Long else
    # will get the following error
    # RuntimeError: expected scalar type Long but found Float
    self.y = torch.from_numpy(y_train).type(torch.LongTensor)
    self.len = self.X.shape[0]

  def __getitem__(self, index):
    return self.X[index], self.y[index]
  def __len__(self):
    return self.len

def load_data(dataset_name, shot_dir, normalize_data=False):
  data_dir = 'drive/MyDrive/classification_data/'
  #dataset_name = 'PenDigits/'
  #shot_dir = '5-shot/'

  #tiny epsilon value
  epsilon = 1e-8

  train_data = np.load(data_dir + dataset_name + shot_dir + 'X_train.npy')
  train_label = np.load(data_dir + dataset_name + shot_dir + 'y_train.npy')
  test_data = np.load(data_dir + dataset_name + 'X_test.npy')
  test_label = np.load(data_dir + dataset_name + 'y_test.npy')
  if normalize_data:
    train_data = (train_data - train_data.mean(axis=1)[:, None]) / (train_data.std(axis=1)[:, None] + epsilon)
    test_data = (test_data - test_data.mean(axis=1)[:, None]) / (test_data.std(axis=1)[:, None] + epsilon)

  return train_data, train_label, test_data, test_label

import torch.optim as optim

def vanilla_train_model(trainloader, train_label, test_data, test_label, input_size):

  model_resnet = ResNet(input_size = input_size, nb_classes=len(np.unique(train_label)))
  criterion = nn.CrossEntropyLoss()


  runSAM = False
  optimizer = 'sgd'




  if runSAM==False:
    optimizer = torch.optim.SGD(model_resnet.parameters(), lr=lr, momentum=0.9)
    #optimizer = optim.Adam(model_resnet.parameters(), lr=1e-8)
  else:
    base_optimizer = torch.optim.SGD # define an optimizer for the "sharpness-aware" update
    optimizer = SAM(model_resnet.parameters(), base_optimizer, lr=lr, momentum=0.9, rho=rho)


  # ResNet vanilla
  #100 epoch for batch = 1024*
  best_loss = 10000 #smaller is better.
  max_limit = 20
  counter = 0


  model_resnet = model_resnet.cuda()

  criterion = nn.CrossEntropyLoss()

  for epoch in range(nEpoch):  # loop over the dataset multiple times
      running_loss = 0.0
      val_running_loss = 0.0
      for i, data in enumerate(trainloader, 0):
          # get the inputs; data is a list of [inputs, labels]
          inputs, labels = data
          inputs = inputs.cuda()
          labels = labels.cuda()


          # zero the parameter gradients
          optimizer.zero_grad()

          # forward + backward + optimize
          outputs1 = model_resnet(torch.tensor(inputs).transpose(1,2).float())
          outputs = outputs1[0]#1

          labels = torch.squeeze(labels, dim=1)

          loss = criterion(outputs, labels)
          loss.backward()
          optimizer.step()

          # print statistics
          running_loss += loss.item()
          print("Epoch:", epoch+1, "-->", running_loss, loss.item())

  print('Finished Training')


  data = torch.from_numpy(test_data).float()
  data = data.cuda()

  pred, embed = model_resnet(data.transpose(1,2).float())

  correct = 0
  total = 0

  labels = torch.squeeze(torch.from_numpy(test_label), dim=1)
  _, predicted = torch.max(pred.data, 1)
  total = labels.size(0)
  correct = (predicted == labels.cuda()).sum().item()
  acc = correct/total

  print("Final Accuracy: ",acc)

  return acc

def save_to_file_directory(data_dir, dataset_name, shot_dir, normalize_data, acc):
  path = data_dir + dataset_name + shot_dir
  with open(path+'results_vanilla_resnet.txt', 'w') as f:
    f.write(dataset_name + '\n')
    f.write(shot_dir + '\n')
    f.write(str(normalize_data) + '\n')
    f.write(str(acc) + '\n')
    f.write(str(acc.mean()))

def save_to_dataframe(dataframe_name, dataset_name, shot_dir, normalize_data, acc):
  # path to the dataframe csv file
  path = dataframe_name

  # new dataframe for appending (accuracy tensor in mean)
  new_data = {
      'Dataset': [dataset_name],
      'Shots': [shot_dir],
      'Normalization': [normalize_data],
      'Result': [acc.mean()]
  }

  new_df = pd.DataFrame(new_data)

  # append the new dataframe to the existing csv
  new_df.to_csv(path, mode='a', header=False, index=False)

def full_training(dataset_name, shot_dir, normalize_data = True):
  data_dir = 'drive/MyDrive/classification_data/'
  #dataset_name = 'ArticularyWordRecognition/'
  #shot_dir = '5-shot/'


  print('reading ' + dataset_name + '...')
  train_data, train_label, test_data, test_label = load_data(dataset_name, shot_dir, normalize_data)
  print(train_data.shape)
  traindata = Dataset(train_data, train_label)

  input_size = train_data.shape[-1]

  print(input_size)

  batch_size = 1024

  trainloader = DataLoader(traindata, batch_size=batch_size,
                          shuffle=True, num_workers=2)

  acc = []
  for i in range(5):
    acc_tmp = vanilla_train_model(trainloader, train_label, test_data, test_label, input_size)
    print(i)
    acc.append(acc_tmp)

  acc = np.array(acc)

  # save the data
  save_to_file_directory(data_dir, dataset_name, shot_dir, normalize_data, acc)

  # save to dataframe
  save_to_dataframe(filepath, dataset_name, shot_dir, normalize_data, acc)

  return acc

# Build your dataframes in this logic for result saving.
# Replace variables to desired locations.
'''
# columns for our results dataframe
columns = ["Dataset", "Shots", "Normalization", "Result"]

# dataframe construction
df = pd.DataFrame(columns = columns)

# filepath for our csv
filepath = 'drive/MyDrive/classification_data/' + 'results_vanilla_resnet_Character_Traj.csv'

# creating empty df and csv
df.to_csv(filepath, index=False)
'''

# Training cycle example logic:
'''
# change datasets and shot here
dataset_name_vec = ['CharacterTrajectories/']
shot_dir_vec = ['1-shot/']



#dataset_name = 'ArticularyWordRecognition/'
#shot_dir = '10-shot/'

for dataset_name in dataset_name_vec:
  for shot_dir in shot_dir_vec:
    full_training(dataset_name, shot_dir, True)
'''
