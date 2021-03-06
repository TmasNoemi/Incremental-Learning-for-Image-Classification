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
import import_ipynb
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
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib as mlp
import seaborn as sns
from tqdm import tqdm
from modello import resnet32
from seed import seed
sns.set()

class FineTuning(nn.Module):
    def __init__(self, n_classes, dictionary):

        super(FineTuning, self).__init__()

        self.model = resnet32()
        self.prev_model = None
        self.criterion = nn.BCEWithLogitsLoss()
        self.num_classes = n_classes
        self.bs = 10
        self.num_epochs = 70
        self.dictionary = dictionary

    def forward(self,x):
        x = self.model(x) 
        return x


    def train_model(self, dataloader):
        loss_array = []
        self.model = self.model.cuda()
        cudnn.benchmark
        
        start = time.time()
        
        optimizer = optim.SGD(self.model.parameters(), lr=2, momentum=0.9, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [49,63], gamma=0.2)

        for epoch in range(self.num_epochs):
            print('Epoch {}/{}'.format(epoch+1, self.num_epochs))
            print('-' * 10)

            self.model.train() 

            running_loss = 0.0
            running_corrects = 0
            current_step = 0
            for images, labels in dataloader:

                labels_map = [self.dictionary[label.item()] for label in labels]
                
                labels_map = torch.as_tensor(labels_map)
                
                images = images.cuda()
                labels_map = labels_map.cuda()
                
                optimizer.zero_grad()
                
                outputs = self.model(images)
                one_hot = nn.functional.one_hot(labels_map, self.num_classes)
                
                one_hot = one_hot.type_as(outputs)       
                _, preds = torch.max(outputs.data, 1)
                
                loss = self.criterion(outputs, one_hot)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * images.size(0)
                running_corrects += torch.sum(preds == labels_map.data).data.item()
                current_step+=1

            del images, labels
            torch.cuda.empty_cache()

            epoch_loss = running_loss / float(len(dataloader.dataset))
            epoch_acc = running_corrects/ float(len(dataloader.dataset))
            loss_array.append(epoch_loss)
            scheduler.step()

            print('Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))
            print()

        time_elapsed = time.time() - start

        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))


    def test_model(self, dataloader):
        self.model = self.model.cuda()
        self.model.train(False)
        running_corrects = 0
        current_step = 0

        corrects = []
        predictions = []
        for images, labels in tqdm(dataloader):

            labels_map = [self.dictionary[label.item()] for label in labels]

            labels_map = torch.as_tensor(labels_map)
            images = images.cuda()
            labels_map = labels_map.cuda()     
            outputs = self.model(images)
            _, preds = torch.max(outputs.data, 1)
            running_corrects += torch.sum(preds == labels_map.data).data.item()
            for lab,pred in zip(labels_map, preds):
                corrects.append(lab.item())
                predictions.append(pred.item())

            current_step+=1

        accuracy = running_corrects / float(len(dataloader.dataset))
        print('Test Accuracy: {}'.format(accuracy))
        confmat = confusion_matrix(corrects, predictions)
        self.print_confusion_matrix(confmat, set(corrects))
        torch.cuda.empty_cache()
        plt.show()
        return accuracy

    def update_model(self):
        self.prev_model = copy.deepcopy(self.model)
        self.model.fc = nn.Linear(64, self.num_classes+self.bs)  
        self.model.fc.weight.data[0: self.num_classes] = self.prev_model.fc.weight.data
        self.model.fc.bias.data[0: self.num_classes] = self.prev_model.fc.bias.data

    
    def print_confusion_matrix(self,confusion_matrix, class_names):

        df_cm = pd.DataFrame(confusion_matrix, index=class_names, columns=class_names)
        fig = plt.figure(figsize=(6,4))
        try:
            cmap = plt.get_cmap('jet')
            new_cmap = self.truncate_colormap(cmap, 0.05, 0.85)
            heatmap = sns.heatmap(df_cm, annot=False, fmt="d", cmap = new_cmap)
        except ValueError:
            raise ValueError("Confusion matrix values must be integers.")
        heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=5)
        heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=5)
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()
        return fig


    def truncate_colormap(self,cmap, minval=0.0, maxval=1.0, n=100):
        new_cmap = mlp.colors.LinearSegmentedColormap.from_list(
            'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
            cmap(np.linspace(minval, maxval, n)))
        return new_cmap

train_transform=transforms.Compose([
            transforms.RandomCrop(32, padding = 4),
            transforms.RandomHorizontalFlip(p=0.5),           
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR100(root='./data', train=True,
                                        download=True, transform=train_transform)

testset = torchvision.datasets.CIFAR100(root='./data', train=False,
                                       download=True, transform=test_transform)

#generating random indices
samples = seed

#dictionary for mapping the classes
d = {samples[i]:i for i in range(len(samples))}

#creating the batches of data for different groups of 10 classes
batches_train = []
batches_test = []

for i in range(10,110,10):
  x = samples[i-10:i]
  y = samples[0:i]
  idx_train = [i for i,el in enumerate(trainset.targets) if el in x]
  batches_train.append(Subset(trainset,idx_train))
  idx_test = [j for j, el in enumerate(testset.targets) if el in y]
  batches_test.append(Subset(testset, idx_test))


train_dataloaders = [DataLoader(batch, batch_size=128, shuffle=False, num_workers=4) for batch in batches_train]
test_dataloaders = [DataLoader(batch, batch_size=128, shuffle=True, num_workers=4) for batch in batches_test]

cum_acc = np.zeros((10,3))
runs = 3
for run in range(runs):
    finetuning = FineTuning(10, d)
    for i,idx in enumerate(range(10,100,10)):
        print(f'step: {i} classes: {idx}')
        print()
        finetuning.train_model(train_dataloaders[i])
        acc = finetuning.test_model(test_dataloaders[i])
        finetuning.update_model()
        finetuning.num_classes = idx+10
        cum_acc[i, run] = acc

sns.set()
r = np.random.rand(10,3)
means = np.mean(r, axis=1)
std = np.std(r, axis=1)
# cum_acc_lwf = np.random.rand(10,3)
cum_acc_fin = cum_acc
# cum_acc_icarl = np.random.rand(10,3)


sns.set(font_scale=1.2)
sns.set_style("whitegrid")

x = np.arange(10,110,10)
# y1 = np.mean(cum_acc_lwf, axis =1)
# err1 = np.std(cum_acc_lwf, axis = 1)
# print(err1, y1)
y2 = np.mean(cum_acc_fin, axis =1)
err2 = np.std(cum_acc_fin, axis = 1) 

# y3 = np.mean(cum_acc_icarl, axis =1)
# err3 = np.std(cum_acc_icarl, axis = 1)

fig, ax = plt.subplots(figsize = (15,10),)
ax.set_title('Results', fontweight = 'bold')
# ax.errorbar(x , y1, yerr=err1, label = 'Fine Tuning')
ax.errorbar(x , y2, yerr=err2, label = 'Fine Tuning')
# ax.errorbar(x , y3, yerr=err3, label = 'iCaRL')

ax.legend()
ax.set_xlim(min(x)-1, max(x)+1)
ax.yaxis.grid(True) # Hide the horizontal gridlines
ax.xaxis.grid(False) # Show the vertical gridlines
plt.figure()

true_ = [0.87, 0.45, 0.30, 0.22, 0.18, 0.15, 0.12, 0.11, 0.10, 0.09]
x = np.arange(10,110,10)
y = np.mean(cum_acc, axis = 1)
# plt.style.use('dark_background')
fig, ax = plt.subplots()
ax.set_title('Finetuning results', fontweight = 'bold')
ax.plot(x, y, '-o', color = 'dodgerblue')
ax.plot(x, true_, '-*', color = 'red', label = 'target')
ax.grid(b = True)
ax.legend()
ax.set_xlim(min(x), max(x))
plt.figure()


for i, txt in enumerate(y):
  ax.annotate("{:.2f}".format(txt), (x[i], y[i]))
