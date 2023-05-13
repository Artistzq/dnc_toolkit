import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from ..datasets import wrapper as data_wrapper
from ..datasets.selector import Selector, UncertaintyBasedSelector


class TinyTrainer():
  def __init__(self, model):
    self.model = model
  
  def train(self, train_loader, val_loader, num_epochs, lr):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    self.model = self.model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=5e-4)
    
    for epoch in range(num_epochs):
      train_loss = 0.0
      val_loss = 0.0
      train_correct = 0
      train_total = 0
      
      self.model.train()
      for i, (inputs, labels) in enumerate(train_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        outputs = self.model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs.data, 1)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()
        
      train_loss /= len(train_loader.dataset)
      train_accuracy = train_correct / train_total
      
      self.model.eval()
      val_correct = 0
      val_total = 0
      with torch.no_grad():
        for i, (inputs, labels) in enumerate(val_loader):
          inputs = inputs.to(device)
          labels = labels.to(device)
          
          outputs = self.model(inputs)
          loss = criterion(outputs, labels)
          
          val_loss += loss.item() * inputs.size(0)
          _, predicted = torch.max(outputs.data, 1)
          val_total += labels.size(0)
          val_correct += (predicted == labels).sum().item()
          
      val_loss /= len(val_loader.dataset)
      val_accuracy = val_correct / val_total
      
      print("Epoch {:>2} - Train Loss: {:.4f} - Val Loss: {:.4f} - Train Accuracy: {:.2f}% - Val Accuracy: {:.2f}%".format(epoch+1, train_loss, val_loss, train_accuracy*100, val_accuracy*100))
  
  def eval(self, test_loader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    self.model = self.model.to(device)

    correct = 0
    total = 0

    self.model.eval()
    with torch.no_grad():
      for i, (inputs, labels) in enumerate(test_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = self.model(inputs)
        _, predicted = torch.max(outputs.data, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    accuracy = correct / total
    print("Test Accuracy: {:.2f}%".format(accuracy * 100))

