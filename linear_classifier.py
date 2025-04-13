import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, TensorDataset
import numpy as np
from sklearn.metrics import confusion_matrix
from collections import Counter
#import matplotlib.pyplot as plt
#import seaborn as sns

def get_class_indices(labels):
  classes, class_indices = np.unique(labels, return_inverse=True)
  return class_indices


def print_label_count(labels):
  unique_labels, counts = np.unique(labels, return_counts=True)
  
  # Print the results
  for label, count in zip(unique_labels, counts):
      print(f"{label}: {count}")
      
def get_label_counts(dataset):
    labels = [sample[1].item() for sample in dataset]
    counts = Counter(labels)
    return counts
     
class linear_classifier(nn.Module):

  def __init__(self, input_dim=122880, output_dim=3):
    super().__init__()
    self.fc = nn.Linear(input_dim, output_dim)

  def forward(self, x):
    return self.fc(x)
      

def get_validation_loss(model, val_loader):
  model.eval()
  average_loss = 0
  with torch.no_grad():
    for x_batch, y_batch in val_loader:
      x_batch = x_batch.to("cuda")
      y_batch = y_batch.to("cuda")
      outputs = model(x_batch)
      loss = criterion(outputs, y_batch)
      average_loss = loss.item()
  
  
  return average_loss / len(val_loader)


def run_inference(model, test_loader):
    model.eval() 
    
    predicted_labels = []  
    true_labels = [] 
    
    with torch.no_grad():  
      for x_batch, y_batch in test_loader:
        x_batch, y_batch = x_batch.to("cuda"), y_batch.to("cuda") 
        
        outputs = model(x_batch) 
        
        _, predicted = torch.max(outputs, 1)
        
        predicted_labels.append(predicted.cpu().numpy()) 
        true_labels.append(y_batch.cpu().numpy()) 
    
    predicted_labels = np.concatenate(predicted_labels) 
    true_labels = np.concatenate(true_labels)  
    
    result = np.stack((predicted_labels, true_labels), axis=1)
    
    return result
    
      
concept_latent_activations = np.load("/scratch/s3799042/data/Chess_SAE/concept_latent.npy", allow_pickle=True)


X = np.array([sample[4] for sample in concept_latent_activations])  
X = torch.from_numpy(X).float()

Y = np.array([sample[0] for sample in concept_latent_activations])
print_label_count(Y)

Y = get_class_indices(Y)
Y = torch.tensor(Y).long()
flattend_X = X.view(X.size(0), -1)

dataset = TensorDataset(flattend_X, Y)

train_size = int(0.7 * len(dataset))
val_size = int(0.05 * len(dataset))
test_size = len(dataset) - train_size - val_size 

train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

train_counts = get_label_counts(train_dataset)
val_counts = get_label_counts(val_dataset)
test_counts = get_label_counts(test_dataset)

print("Train label counts:", train_counts)
print("Validation label counts:", val_counts)
print("Test label counts:", test_counts)

batch_size = 32 

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


model = linear_classifier()

# Because of the class imbalance the classes are weighted differently
weights = torch.tensor([1, 0.3, 0.1])

criterion = nn.CrossEntropyLoss(weight=weights.to("cuda"))

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
model.to("cuda")
model.train()

for epoch in range(20):
  average_loss = 0.0
  for x_batch, y_batch in train_loader:
    x_batch = x_batch.to("cuda")
    y_batch = y_batch.to("cuda")
    outputs = model(x_batch)
    loss = criterion(outputs, y_batch)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    average_loss += loss.item()
    
  average_loss /= len(train_loader)
  loss_val = get_validation_loss(model, val_loader)
  model.train()
  print(f"Epoch {epoch+1}, Train Loss: {average_loss:.4f}, Val Loss: {loss_val:.4f}")
  
predicted_true_labels = run_inference(model, test_loader)
predicted_labels = predicted_true_labels[:, 0]
true_labels = predicted_true_labels[:, 1]

cm = confusion_matrix(true_labels, predicted_labels)

print(cm)

cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

print("\nConfusion Matrix (Percentages):")
print(np.round(cm_percent, 2))





  
    