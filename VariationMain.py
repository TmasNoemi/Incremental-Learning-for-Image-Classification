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


from Variation_model import resnet32
from seed import seed
sns.set()

print(torch.cuda.get_device_name())

class iCarlNet(nn.Module):
    def __init__(self, n_classes, dictionary):
        # Network architecture
        super(iCarlNet, self).__init__()

        #Main net
        self.model = resnet32()

        self.dictionary = dictionary
        #previous model to compute the distillation loss

        #in feature extractor is removed the last fc layer of resnet32
        self.feature_extractor = nn.Sequential(*(list(self.model.children())[:-2]))
        self.cuda()
        self.compute_means = True
        self.exemplar_means = []

        self.exemplars = []

        # Learning method
        self.optimizer = optim.SGD(self.model.parameters(), lr=2, momentum=0.9, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer,[49,63], gamma=0.2)
        self.criterion_class = nn.BCEWithLogitsLoss()
        self.criterion_dist = nn.MSELoss()

        self.num_epochs = 70     
  
    def forward(self, x, distilled_fc = None):
        x = self.model(x, distilled_fc) 
        return x

    def train_initialize(self, dataloader):
      print('Initialization...')    
      cudnn.benchmark     
      start = time.time()
      optimizer = self.optimizer
      scheduler = self.scheduler
    
      for epoch in range(self.num_epochs):
          print('Epoch {}/{}'.format(epoch+1, self.num_epochs))
          print('-' * 10)

          running_loss = 0.0
          running_corrects = 0.0

          self.model.train()
          for images, labels in dataloader:            
                labels_map = [self.dictionary[label.item()] for label in labels]
                labels_map = torch.as_tensor(labels_map)   
                images = images.to(DEVICE)
                labels_map = labels_map.to(DEVICE)
                
                optimizer.zero_grad()

                outputs = self.forward(images)
                one_hot = nn.functional.one_hot(labels_map, 10)
                one_hot = one_hot.type_as(outputs)
                
                _, preds = torch.max(outputs.data, 1)
                
                loss = self.criterion_class(outputs, one_hot)

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

      print('Copying weights on fully connected for classes 0 - 10:')
      self.model.hidden[0].weight.data = self.model.fc.weight.data[:10]
      self.model.hidden[0].bias.data = self.model.fc.bias.data[:10]
      self.model.hidden[0].requires_grad_ = False
      
      time_elapsed = time.time() - start
      print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    def update_model(self, step):
      old_weights = self.model.fc.weight.data
      old_bias = self.model.fc.bias.data

      self.model.fc = nn.Linear(64, step+10)

      self.model.fc.weight.data[:step] = old_weights
      self.model.fc.bias.data[:step] = old_bias

    def train_fully_connected(self, dataloader, step):
      cudnn.benchmark 
      specialized_model = resnet32()
      print()
      print(f'Train fully connected related to classes {step} : {step+10}')
      print()
      print(f'fc index: {step//10}')

      specialized_model = specialized_model.cuda()
      start = time.time()

      optimizer = optim.SGD(specialized_model.parameters(), lr=2, momentum=0.9, weight_decay=1e-5)
      scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [49,63], gamma=0.2)
      
      list_of_lab = []
      for _, labels in dataloader:
            list_of_lab += labels.numpy().tolist()

      samples = list(set(list_of_lab))
      d = {samples[i]:i for i in range(len(samples))}

      for epoch in range(self.num_epochs):
          print('Epoch {}/{}'.format(epoch+1, self.num_epochs))
          print('-' * 10)

          running_loss = 0.0
          running_corrects = 0.0

          self.model.train()
          for images, labels in dataloader:            
                labels_map = [d[label.item()] for label in labels]
                labels_map = torch.as_tensor(labels_map)   
                images = images.cuda()
                labels_map = labels_map.cuda()
                
                optimizer.zero_grad()              
                outputs = specialized_model.forward(images)

                one_hot = nn.functional.one_hot(labels_map, 10)
                one_hot = one_hot.type_as(outputs)
                
                _, preds = torch.max(outputs.data, 1)
                
                loss = self.criterion_class(outputs, one_hot)

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
      print(f'Copia fully connected pesi per classi da {step} a {step+10} ')

      self.model.hidden[(step)//10].weight.data = specialized_model.fc.weight.data
      self.model.hidden[(step)//10].bias.data = specialized_model.fc.bias.data
      self.model.hidden[(step)//10].requires_grad_ = False

      time_elapsed = time.time() - start
      print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
          
    def update_feature_extractor(self):

      self.feature_extractor = copy.deepcopy(self.model)
      self.feature_extractor = nn.Sequential(*list(self.feature_extractor.children())[:-2])

    def reduce_exemplar_sets(self, m):
        for y, P_y in enumerate(self.exemplars):
            self.exemplars[y] = P_y[:m]

    def combine_dataset_with_exemplars(self, batches_train, mini_batches):
      dataset = batches_train
      for i,exemplars_indices in enumerate(self.exemplars):
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
          # feature = torch.softmax(feature, dim=1)  
            
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
      # print(len(self.exemplars))

    def train(self, dataloader, step):

      self.compute_means = True
      self.model.cuda()
      cudnn.benchmark 
      start = time.time()

      self.model.train() 
      optimizer = optim.SGD(self.model.parameters(), lr=2, momentum=0.9, weight_decay=1e-5)
      scheduler = optim.lr_scheduler.MultiStepLR(optimizer,[49,63], gamma=0.2)

      for epoch in range(self.num_epochs):
            print('Epoch {}/{}'.format(epoch+1, self.num_epochs))
            print('-' * 10)

            running_loss = 0.0
            running_corrects = 0.0

            for images, labels in dataloader:            
                  labels_map = [self.dictionary[label.item()] for label in labels]
                  labels_map = torch.as_tensor(labels_map)   
                  images = images.cuda()
                  labels_map = labels_map.cuda()
                  
                  optimizer.zero_grad()

                  tot_outputs = [] 
                  I = []
                  for r in range(0,(step//10)):  
                    index = [i for i,lab in enumerate(labels_map) if lab in range(r*10,r*10+10)] 
                    I.append(index)                   
                    if not index:
                      tot_outputs.append([])
                      continue
                    else:  
                      old_outputs = self.forward(images[index], r)
                      tot_outputs.append(old_outputs)

                  outputs = self.forward(images)  
                  OUTPUTS = []
                  TARGETS = []

                  for j,idx in enumerate(I):
                    if not idx:
                      continue
                    else:
                      OUTPUTS.append(outputs[idx][:,j*10:j*10+10])
                      TARGETS.append(tot_outputs[j])
                  
                  TARGETS = torch.cat((TARGETS),0).squeeze()
                  OUTPUTS = torch.cat((OUTPUTS),0).squeeze()

                  TARGETS = TARGETS.cuda()
                  OUTPUTS = OUTPUTS.cuda()

                  one_hot = nn.functional.one_hot(labels_map, step + 10)
                  one_hot = one_hot.type_as(outputs)
                  
                  weight = step/(step+10)
                  _, preds = torch.max(outputs.data, 1)

                  loss_dist = self.criterion_dist(torch.sigmoid(OUTPUTS), torch.sigmoid(TARGETS))

                  loss_class = self.criterion_class(outputs, one_hot)
                  weight = (step/(step+10))*0.1
                  loss = loss_dist* weight + loss_class*(1-weight)

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

      self.update_feature_extractor()
      time_elapsed = time.time() - start
      print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    def NCM(self, images):

      self.feature_extractor.train(False)
      phi_imgs = self.feature_extractor(images)
      
      if self.compute_means:
        exemplar_means = []
        for i,list_of_ex in enumerate(self.exemplars):
          ex_features = []
          for j in list_of_ex:
            img = mini_batches[i].__getitem__(j)        
            with torch.no_grad():
              exemplar = Variable(img[0]).cuda()
              phi_ex = self.feature_extractor(exemplar.unsqueeze(0))
              phi_ex = phi_ex.squeeze()
              phi_ex.data = phi_ex.data / phi_ex.data.norm() # Normalize
              ex_features.append(phi_ex)

          ex_features = torch.stack(ex_features)
          mu_y = ex_features.mean(0).squeeze()
          mu_y.data = mu_y.data / mu_y.data.norm() # Normalize
          exemplar_means.append(mu_y)

        self.exemplar_means = exemplar_means
        self.compute_means = False

      exemplar_means = self.exemplar_means

      means = torch.stack(exemplar_means) 

      means = torch.stack([means] * 128) 
      means = means.transpose(1, 2) 
      
      for i in range(phi_imgs.size(0)): 
        phi_imgs.data[i] = phi_imgs.data[i] / phi_imgs.data[i].norm()

      phi_imgs = phi_imgs.squeeze(2) 
      phi_imgs = phi_imgs.expand_as(means)
      
      dists = (phi_imgs - means).pow(2).sum(1).squeeze() 
      _, preds = dists.min(1)
      
      return preds

    def Classify(self, dataloader): 
        total = 0.0
        correct = 0.0
        corrects = []
        predictions = []
        for images, labels in dataloader:
            labels_map = [self.dictionary[label.item()] for label in labels]
                
            labels_map = torch.as_tensor(labels_map)
            images = Variable(images).cuda()
            preds = icarl.NCM(images)
            total += labels.size(0)
            for lab,pred in zip(labels_map, preds):
                corrects.append(lab.item())
                predictions.append(pred.item())

            correct += (preds.data.cpu() == labels_map).sum()
        
        confmat = confusion_matrix(corrects, predictions)
        print_confusion_matrix(confmat, set(corrects))
        plt.show()
        accuracy = correct / total
        print(f'Test Accuracy: {accuracy:.4f}')
        return accuracy

def print_confusion_matrix(confusion_matrix, class_names, figsize = (6,4), fontsize=5):

    df_cm = pd.DataFrame(
        confusion_matrix, index=class_names, columns=class_names, 
    )
    fig = plt.figure(figsize=figsize)
    try:
        cmap = plt.get_cmap('jet')
        new_cmap = truncate_colormap(cmap, 0.05, 0.85)
        heatmap = sns.heatmap(df_cm, annot=False, fmt="d", cmap = new_cmap)
    except ValueError:
        raise ValueError("Confusion matrix values must be integers.")
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
    return fig


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = mpl.colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

#Define datasets
train_transform=transforms.Compose([
            
            transforms.RandomCrop(32, padding = 4),
            transforms.RandomHorizontalFlip(p=0.5),            
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

test_transform = transforms.Compose([
            transforms.CenterCrop(32),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR100(root='./data', train=True,
                                        download=True, transform=train_transform)

testset = torchvision.datasets.CIFAR100(root='./data', train=False,
                                       download=True, transform=test_transform)
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

"""**Initialization of iCarl...**"""

icarl = iCarlNet(10,d)
icarl.train_initialize(train_dataloaders[0])
K = 2000
m = K//10
#initialization of exemplars for the first batch

for y in range(10):
  icarl.create_exemplars(mini_batches[y],m)
print(f'done now there are {len(icarl.exemplars)} exemplars')
test_accuracy = icarl.Classify(test_dataloaders[0])

"""**Train iCarl over all the other classes...**"""

Accuracy = [test_accuracy] 
#initialize the accuracy list with the first test accuracy
indice = 1
for idx in range(10,100,10):
  
  icarl.update_model(idx) #increment the outputs in the fc
  
  dataloader = icarl.combine_dataset_with_exemplars(batches_train[indice], mini_batches) #update the network with backprop

  icarl.train(dataloader,idx)

  m = int(K/(idx+10))
  icarl.reduce_exemplar_sets(m)

  # Construct exemplar sets for new classes
  print(f"creating exemplars for idx = {idx}")
  for y in range(idx,idx+10):
    icarl.create_exemplars(mini_batches[y], m)
  print(f'done now there are {len(icarl.exemplars)} exemplars')
  print(f"CLASSES {idx+10}")

  test_accuracy = icarl.Classify(test_dataloaders[indice])
  icarl.train_fully_connected(train_dataloaders[indice], idx)
  Accuracy.append(test_accuracy)
  indice +=1

