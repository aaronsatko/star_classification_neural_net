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


EPOCHS = 11

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# load dataset
data = pd.read_csv('star_classification_pruned.csv')

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

# nn architecture
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # First fully connected layer
        self.fc1 = nn.Linear(6, 128)
        # Second fully connected layer
        #self.fc2 = nn.Linear(128, 128)
        self.fc2 = nn.Linear(128, 64)
        # Output layer, 3 classes
        self.fc3 = nn.Linear(64, 3)

    def forward(self, x):
        # ReLU activation function after first layer
        x = F.relu(self.fc1(x))
        # ReLU activation function after second layer
        x = F.relu(self.fc2(x))
        #x = F.relu(self.fc3(x))
        # No activation for out
        x = self.fc3(x)
        return x
    
    def extract_features(self, x):
        # Extract features from the second layer
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x
    
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
for epoch in range(EPOCHS):
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

    net.eval()  # Set the network to evaluation mode
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

    print(f'Epoch {epoch + 1}, Training Loss: {avg_train_loss}, Validation Loss: {avg_val_loss}')

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
torch.save(net.state_dict(), 'model.pth')


import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.manifold import TSNE
import seaborn as sns


# loss_curve
epochs = range(1, EPOCHS + 1)
plt.figure(figsize=(10, 6))
plt.plot(epochs, training_losses, label='Training Loss', marker='o')
plt.plot(epochs, validation_losses, label='Validation Loss', marker='o')
plt.title('Loss Curve')
plt.xlabel('Epochs')
plt.ylabel('Loss')
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


# TSNE Plot
tsne = TSNE(n_components=2, random_state=42)
X_reduced = tsne.fit_transform(X_test)
plt.figure(figsize=(10, 8))
plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y_test.numpy(), cmap="jet")
plt.colorbar()
plt.title("Pre-Classification T-Distributed Stochastic Neighbor Embedding")
plt.savefig('plots/tsne.png')


# Extract features for t-SNE
net.eval()
features = []
with torch.no_grad():
    for inputs, _ in test_loader:
        inputs = inputs.to(device)
        feature = net.extract_features(inputs)
        features.extend(feature.cpu().numpy())

# Apply t-SNE on the extracted features
tsne_features = TSNE(n_components=2, random_state=42).fit_transform(np.array(features))

# Plot post-classification t-SNE
plt.figure(figsize=(10, 8))
plt.scatter(tsne_features[:, 0], tsne_features[:, 1], c=y_test.numpy(), cmap="jet")
plt.colorbar()
plt.title("Post-Classification T-Distributed Stochastic Neighbor Embedding")
plt.savefig('plots/post_tsne.png')