import torch
import torch.nn as nn
# neural net module
import torch.nn.functional as F
# convolution pooling, activation functio, loss
import torch.optim as optim
# optimization algos
from torch.utils.data import DataLoader, TensorDataset
# data handling
import pandas as pd
from sklearn.model_selection import train_test_split
# split dataset into training and testing sets
from sklearn.preprocessing import StandardScaler
# preprocessing
from sklearn.metrics import accuracy_score
# calculate accuracy of a model
import numpy as np
# progress bar
from tqdm import tqdm
import joblib
from model import Net


EPOCHS = 11

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# load dataset
data = pd.read_csv('star_classification.csv')

# preprocess data in a hashmap  to convert classes to numerical
class_map = {'GALAXY': 0, 'STAR': 1, 'QSO': 2}
data['class'] = data['class'].map(class_map)

# input data 
X = data[['u', 'g', 'r', 'i', 'z', 'redshift']].values
# output data
y = data['class'].values

# training, testing, and validation sets
# Split data into training (80%) and testing + validation (20%)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
# split data into validation (10%) and testing (10%)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)


# standardization of features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

joblib.dump(scaler, 'model\scaler.save')

# convert data into torch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
X_val = torch.tensor(X_val, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_val = torch.tensor(y_val, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)


# Create tensor datasets
train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)
test_dataset = TensorDataset(X_test, y_test)

# batch handling
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)



net = Net().to(device)

# loss function
loss_func = nn.CrossEntropyLoss()

# use adam optimizer
optimizer = optim.Adam(net.parameters(), lr=0.001)


# train the neural net
# Lists to store losses
training_losses = []
validation_losses = []

# train the neural net
for epoch in tqdm(range(EPOCHS), desc="Training Progress"):
    net.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = loss_func(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # Calculate average training loss for this epoch
    avg_train_loss = running_loss / len(train_loader)
    training_losses.append(avg_train_loss)

    net.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            loss = loss_func(outputs, labels)
            val_loss += loss.item()

    # Calculate average validation loss for this epoch
    avg_val_loss = val_loss / len(val_loader)
    validation_losses.append(avg_val_loss)

    print(f'Epoch {epoch + 1}/{EPOCHS}: Training Loss = {avg_train_loss:.4f}, Validation Loss = {avg_val_loss:.4f}')

print('Finished Training')


# Test net
net.eval()
y_pred = []
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = net(inputs)
        _, predicted = torch.max(outputs.data, 1)
        y_pred.extend(predicted.cpu().numpy())
        
# accuracy
accuracy = accuracy_score(y_test.numpy(), np.array(y_pred))
print(f'Accuracy: {accuracy * 100:.2f}%')

# save results for reuse
torch.save(net.state_dict(), 'model\model.pth')


import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns


# loss_curve
epochs = range(1, EPOCHS + 1)
plt.figure(figsize=(10, 6))
plt.plot(epochs, training_losses, label='Training Loss', marker='o', color='blue')
plt.plot(epochs, validation_losses, label='Validation Loss', marker='o', color='red')
plt.title('Training and Validation Loss Over Epochs', fontsize=14)
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.legend()
plt.grid(True)
plt.savefig('plots/loss_curve.png')


# confusion matrix
cm = confusion_matrix(y_test.numpy(), np.array(y_pred))
plt.figure(figsize=(10, 8))
class_names = ['GALAXY', 'STAR', 'QSO']
sns.heatmap(cm, annot=True, fmt="d", xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix')
plt.ylabel('Actual Class')
plt.xlabel('Predicted Class')
plt.savefig('plots/confusion_matrix.png')

