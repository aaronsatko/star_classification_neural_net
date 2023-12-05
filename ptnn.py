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


# load dataset
data = pd.read_csv('star_classification_pruned.csv')

# preprocess data in a hashmap  to convert classes to numerical
class_map = {'GALAXY': 0, 'STAR': 1, 'QSO': 2}
data['class'] = data['class'].map(class_map)

# input data 
X = data[['u', 'g', 'r', 'i', 'z', 'redshift']].values
# output data
y = data['class'].values

# training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# standardization of features
scalar = StandardScaler()
X_train = scalar.fit_transform(X_train)
X_test = scalar.transform(X_test)

# convert data into torch tensors

X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)


# Create tensor datasets
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

# batch handling
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# nn architecture
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # First fully connected layer
        self.fc1 = nn.Linear(6, 128)  # 6 input features
        # Second fully connected layer
        self.fc2 = nn.Linear(128, 64)
        # Output layer, 3 classes
        self.fc3 = nn.Linear(64, 3)

    def forward(self, x):
        # Apply ReLU activation function after first layer
        x = F.relu(self.fc1(x))
        # Apply ReLU activation function after second layer
        x = F.relu(self.fc2(x))
        # No activation needed in the output layer
        x = self.fc3(x)
        return x
    
net = Net()

# loss function
loss_func = nn.CrossEntropyLoss()

# use adam optimizer
optimizer = optim.Adam(net.parameters(), lr=0.001)


# train the neural net
for epoch in range(10):  # loop over the dataset multiple times
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = loss_func(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    print(f'Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)}')

print('Finished Training')


# test net
net.eval()
y_pred = []
with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = net(inputs)
        _, predicted = torch.max(outputs.data, 1)
        y_pred.extend(predicted.numpy())
        
        
# accuracy
accuracy = accuracy_score(y_test.numpy(), np.array(y_pred))
print(f'Accuracy: {accuracy * 100:.2f}%')






