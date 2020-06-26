# -*- coding: utf-8 -*-

import os
import logging
import torch
import torchvision
import numpy as np
import pandas as pd
import copy
import os.path
import sys
import random
import time
from PIL import Image
from tqdm import tqdm

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.nn.init as init
from torch.utils.data import Subset, DataLoader
from torch.backends import cudnn
from torch.autograd import Variable
from torchvision.datasets import VisionDataset
from torchvision import transforms, datasets

from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns


from model import resnet32
from seed import seed
from mlp import MLP
sns.set()

class iCarlNet(nn.Module):
    def __init__(self, n_classes, dictionary):
        # Network architecture
        super(iCarlNet, self).__init__()
        
        self.n_classes = n_classes
        self.bs = 10
        self.dictionary = dictionary
        #Main net
        self.model = resnet32()

        #previous model to compute the distillation loss
        self.prev_model = copy.deepcopy(self.model)

        #in feature extractor is removed the last fc layer of resnet32
        self.feature_extractor = nn.Sequential(*(list(self.model.children())[:-1]))

        self.cuda()
        self.exemplars = []

        # Learning method
        self.criterion = nn.BCEWithLogitsLoss()
        self.num_epochs = 70
        
        self.ex_features = []
        self.ex_labels = []

    def update_prev_model(self):
      self.prev_model = copy.deepcopy(self.model)

    def update_model(self):
      self.model.fc = nn.Linear(64, self.n_classes)  
      self.model.fc.weight.data[0: self.n_classes - self.bs] = self.prev_model.fc.weight.data
      self.model.fc.bias.data[0: self.n_classes - self.bs] = self.prev_model.fc.bias.data

    def update_feature_extractor(self):
      self.feature_extractor = copy.deepcopy(self.model)
      self.feature_extractor = nn.Sequential(*list(self.feature_extractor.children())[:-1])

    def forward(self, x):
        x = self.model(x) 
        return x

    def train_initialize(self, dataloader):      
      cudnn.benchmark     
      start = time.time()
      
      optimizer = optim.SGD(self.model.parameters(), lr=2, momentum=0.9, weight_decay=1e-5)
      scheduler = optim.lr_scheduler.MultiStepLR(optimizer,[49,63], gamma=0.2)

      for epoch in range(self.num_epochs):
          print('Epoch {}/{}'.format(epoch+1, self.num_epochs))
          print('-' * 10)

          running_loss = 0.0
          running_corrects = 0.0

          self.model.train()
          for images, labels in dataloader:            
                labels_map = [self.dictionary[label.item()] for label in labels]
                labels_map = torch.as_tensor(labels_map)   
                images = images.cuda()
                labels_map = labels_map.cuda()
                
                optimizer.zero_grad()

                outputs = self.forward(images)
                one_hot = nn.functional.one_hot(labels_map, self.n_classes)
                one_hot = one_hot.type_as(outputs)
                
                _, preds = torch.max(outputs.data, 1)                
                loss = self.criterion(outputs, one_hot)

                loss.backward()
                optimizer.step()

                running_corrects += torch.sum(preds == labels_map.data).data.item()
                running_loss += loss.item() * images.size(0)

          scheduler.step()
          del images,labels
          torch.cuda.empty_cache()

          epoch_loss = running_loss/float(len(dataloader.dataset))
          epoch_acc = running_corrects/float(len(dataloader.dataset))

          print('Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))
          print()

      #after each training the feature extractor has to be updated with the same weights as the trained network
      self.update_feature_extractor()
      self.update_prev_model()

      time_elapsed = time.time() - start
      print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    def reduce_exemplar_sets(self, m):
        for y, P_y in enumerate(self.exemplars):
            self.exemplars[y] = P_y[:m]

    def combine_dataset_with_exemplars(self, batches_train, mini_batches):
      dataset = batches_train
      for i, exemplars_indices in enumerate(self.exemplars):
        dataset = torch.utils.data.ConcatDataset([dataset, Subset(mini_batches[i],exemplars_indices)])
      dataloader = DataLoader(dataset, batch_size = 128, shuffle = True, drop_last = True, num_workers = 4)
      return dataloader

    def create_exemplars(self, mini_batch, m):

      features = []
      self.feature_extractor.train(False)
      for image, label in mini_batch:
        with torch.no_grad():
          x = Variable(image).cuda()         
          feature = self.feature_extractor(x.unsqueeze(0))         
          feature = feature.squeeze().data.cpu().numpy()
          feature = feature / np.linalg.norm(feature)          
          features.append(feature)

      features = np.array(features)
      class_mean = np.mean(features, axis=0)
      class_mean = class_mean / np.linalg.norm(class_mean)

      exemplar_set = []
      exemplar_features = []

      for k in range(m):
          S = np.sum(exemplar_features, axis=0)
          phi = features
          mu = class_mean #mu is an array of shape (64,)
          mu_p = 1.0/(k+1) * (phi + S) #mu_p is an array of shape (500,64)
          mu_p = mu_p / np.linalg.norm(mu_p)
          res = np.sqrt(np.sum((mu - mu_p) ** 2, axis=1))

          for i in exemplar_set:            
            res[i] = 10000*res[i] #solves the problem of selecting more than one time the same image         
          i = np.argmin(res) #broadcast between mu and mu_p before the argmin the result is a vector of shape (500,)
          exemplar_set.append(i)
          exemplar_features.append(features[i])

      self.exemplars.append(exemplar_set)

    def train(self, dataloader):

      self.compute_means = True
      self.model.to('cuda')
      cudnn.benchmark 
      start = time.time()

      self.model.train() 
      self.prev_model.train(False)

      optimizer = optim.SGD(self.model.parameters(), lr=2, momentum=0.9, weight_decay=1e-5)
      scheduler = optim.lr_scheduler.MultiStepLR(optimizer,[49,63], gamma=0.2)

      for epoch in range(self.num_epochs):
          print('Epoch {}/{}'.format(epoch+1, self.num_epochs))
          print('-' * 10)

          running_loss = 0.0
          running_corrects = 0
          current_step = 0
          for images, labels in dataloader:
              
              labels_map = [self.dictionary[label.item()] for label in labels]
              labels_map = torch.as_tensor(labels_map)
              images = images.cuda()
              labels_map = labels_map.cuda()

              optimizer.zero_grad()
              outputs = self.forward(images)
              old_outputs = self.prev_model(images)

              one_hot = nn.functional.one_hot(labels_map,self.n_classes + self.bs)
              one_hot = one_hot.type_as(outputs)

              _, preds = torch.max(outputs.data, 1)
              _, old_preds = torch.max(old_outputs.data,1)

              old_outputs = torch.sigmoid(old_outputs)
              targets = [old_outputs, one_hot]              
              targets = torch.cat((old_outputs, one_hot[:,self.n_classes:self.n_classes + self.bs]),1)
              loss = self.criterion(outputs, targets)
              loss.backward()
              optimizer.step()

              running_loss += loss.item() * images.size(0)
              running_corrects += torch.sum(preds == labels_map.data).data.item() #accuracy on new classes 

              current_step+=1

          scheduler.step()

          del images, labels
          torch.cuda.empty_cache()
          epoch_loss = running_loss / float(len(dataloader.dataset))
          epoch_acc = running_corrects/ float(len(dataloader.dataset))
        
          print('Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))
          print()

      self.update_feature_extractor()
      self.update_prev_model()

      time_elapsed = time.time() - start
      print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        
    def extract_train_data(self,mini_batches):
      ex_labels = []
      ex_features = []

      for i, list_ex in enumerate(self.exemplars):
        for j in list_ex:
          img = mini_batches[i].__getitem__(j)
          with torch.no_grad():
            exemplar = Variable(img[0]).cuda()
            ex_labels.append(img[1])
            phi_ex = self.feature_extractor(exemplar.unsqueeze(0))
            phi_ex = phi_ex.squeeze()
            phi_ex.data = phi_ex.data / phi_ex.data.norm() # Normalize 
            ex_features.append(phi_ex)

      self.ex_features = torch.stack(ex_features)

      preds_map = [self.dictionary[lab] for lab in ex_labels]
      self.ex_labels = preds_map

    def train_classifier(self, model):
        
      self.ex_labels = torch.as_tensor(self.ex_labels).cuda()

      criterion = nn.BCEWithLogitsLoss()
      optimizer = optim.SGD(model.parameters(), lr = 2, momentum = 0.9, weight_decay = 5e-5)
      scheduler = optim.lr_scheduler.MultiStepLR(optimizer,[450, 600, 750], gamma=0.1)

      model = model.cuda()
      model.train()
      cudnn.benchmark
      
      epochs = 2000
      best_model = copy.deepcopy(model.state_dict())
      best_accuracy = 0.0
      print()
      cum_loss = []
      cum_acc = []
      print('Training MLP \n')
      for epoch in range(epochs):

          optimizer.zero_grad()
          ex_features = Variable(self.ex_features, requires_grad=True)

          # Forward pass
          outputs = model(ex_features)
          one_hot = nn.functional.one_hot(self.ex_labels, self.n_classes)
          one_hot = one_hot.type_as(outputs)

          _, preds = torch.max(outputs.data,1)
          loss = criterion(outputs, one_hot)
          
          loss.backward() 
          optimizer.step()

          corrects = torch.sum(preds == self.ex_labels.data).data.item() 
          epoch_acc = corrects/2000
          cum_loss.append(loss.item()) 
          cum_acc.append(epoch_acc)
            
          if epoch_acc > best_accuracy:
              best_accuracy = epoch_acc
              best_model = copy.deepcopy(model.state_dict())

          scheduler.step()
          if epoch%100 == 0:
            print()
            print('Epoch {}/{}'.format(epoch,epochs))
            print('-'*10)
            print('Loss: {:.4f} Acc: {:.4f}'.format(loss.item(), epoch_acc))

      plt.figure()
      plt.title('MLP Train Loss')
      plt.plot(np.arange(epochs), cum_loss)
      plt.figure()
      plt.title('MLP Train Accuracy')
      plt.plot(np.arange(epochs), cum_acc)
        
      model.load_state_dict(best_model)
      return model


    def Classify(self, dataloader, step =  None, prev_classifier = None):
  
        self.extract_train_data(mini_batches)
            
        classifier = MLP(self.n_classes)

        if prev_classifier is not None and step is not None:
            for i in [1,4]:
                classifier.model[i].weight.data = prev_classifier.model[i].weight.data
                classifier.model[i].bias.data = prev_classifier.model[i].bias.data
            classifier.model[7].weight.data[:step] = prev_classifier.model[7].weight.data
            classifier.model[7].bias.data[:step] = prev_classifier.model[7].bias.data

        classifier = self.train_classifier(classifier)
        classifier.train(False)

        total = 0.0
        correct = 0.0

        corrects = []
        predictions = []
        print()
        print('Testing \n')

        for images, labels in dataloader:

            labels_map = [self.dictionary[label.item()] for label in labels] 
            labels_map = torch.as_tensor(labels_map)

            images = Variable(images).cuda()
            self.feature_extractor.train(False)

            phi_imgs = self.feature_extractor(images)
            
            for i in range(phi_imgs.size(0)): 
                phi_imgs.data[i] = phi_imgs.data[i] / phi_imgs.data[i].norm()
            phi_imgs = phi_imgs.squeeze()
            phi_imgs = phi_imgs.detach()
            
            phi_imgs = Variable(phi_imgs).cuda()

            outputs = classifier(phi_imgs)
            _, preds = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (preds.data.cpu() == labels_map).sum()
            
            for lab,pred in zip(labels_map, preds):
                corrects.append(lab.item())
                predictions.append(pred.item())

        confmat = confusion_matrix(corrects, predictions)
        print_confusion_matrix(confmat, set(corrects))
        plt.show()
        accuracy = correct / total
        print(f'Test Accuracy: {accuracy:.4f}')
        return accuracy, classifier

    
#---------------------------------------------------------------------------------------------------------------------------------    
    
    
def print_confusion_matrix(confusion_matrix, class_names):

    df_cm = pd.DataFrame(confusion_matrix, index=class_names, columns=class_names)
    fig = plt.figure(figsize= (6,4))
    try:
        cmap = plt.get_cmap('jet')
        new_cmap = truncate_colormap(cmap, 0.05, 0.85)
        heatmap = sns.heatmap(df_cm, annot=False, fmt="d", cmap = new_cmap)
    except ValueError:
        raise ValueError("Confusion matrix values must be integers.")
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=5)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=5)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
    return fig


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = mpl.colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap


#---------------------------------------------------------------------------------------------------------------------------------

train_transform=transforms.Compose([
            transforms.RandomCrop(32, padding = 4),
            transforms.RandomHorizontalFlip(p=0.5),            
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR100(root='./data', train=True,download=True, transform=train_transform)

testset = torchvision.datasets.CIFAR100(root='./data', train=False,download=True, transform=test_transform)
DEVICE = 'cuda'

#mini batches will be the base of construction of exemplars
mini_batches = []
for i in seed:
  idx_train = [j for j,el in enumerate(trainset.targets) if el == i]
  mini_batches.append(Subset(trainset,idx_train))

#dictionary for mapping the classes
d = {seed[i]:i for i in range(len(seed))}

#creating the batches of data for different groups of 10 classes
batches_train = []
batches_test = []

for i in range(10,110,10):
  x = seed[i-10:i]
  y = seed[0:i]
  idx_train = [i for i,el in enumerate(trainset.targets) if el in x]
  batches_train.append(Subset(trainset,idx_train))
  idx_test = [j for j, el in enumerate(testset.targets) if el in y]
  batches_test.append(Subset(testset, idx_test))


train_dataloaders = [DataLoader(batch, batch_size=128, shuffle=False, num_workers=4, drop_last = True) for batch in batches_train]
test_dataloaders = [DataLoader(batch, batch_size=128, shuffle=True, num_workers=4, drop_last = True) for batch in batches_test]


#-------------------------------------------------------------------------------------------------------------------------------------

#Inizialization of iCarl on the first set of ten classes
icarl = iCarlNet(10, d)
icarl.train_initialize(train_dataloaders[0])
K = 2000  #memory size 
m = int(K/10) #number of exemplars per class
#initialization of exemplars for the first batch
print('Creation exemplars set...')
for y in range(10):
  icarl.create_exemplars(mini_batches[y], m)
print('done')
test_accuracy = icarl.Classify(test_dataloaders[0])

#------------------------------------------------------------------------------------------------------------------------------------

Accuracy = [test_accuracy] #initialize the accuracy list with the first test accuracy

# Training iCaRL all over the classes 
for i,idx in enumerate(range(10,100,10)):

  icarl.n_classes = idx + 10
  icarl.update_model() #increment the outputs in the fc
  dataloader = icarl.combine_dataset_with_exemplars(batches_train[i+1], mini_batches) #update the network with backpropagation
  icarl.train(dataloader)

  m = int(K/icarl.n_classes) 
  icarl.reduce_exemplar_sets(m)

  # Construct exemplar sets for new classes
  for y in range(idx,idx+10):
      icarl.create_exemplars(mini_batches[y], m)
  print('done')

  test_accuracy = icarl.Classify(test_dataloaders[i+1])
  Accuracy.append(test_accuracy)

