import torch
import torch.nn.functional as F
import joblib
from model import Net

net = Net()
net.load_state_dict(torch.load('model\model.pth'))
net.eval()

scaler = joblib.load('model\scaler.save')

# Function to input new data
def manual_input():
    print("Enter the values for u, g, r, i, z, redshift:")
    values = list(map(float, input().split()))
    return values

# Function to preprocess and predict
def predict_class(input_data, model, scaler):
    # Preprocess the data
    input_data = scaler.transform([input_data])  # Use the same scaler as in training
    input_data = torch.tensor(input_data, dtype=torch.float32)
    
    # Predict
    with torch.no_grad():
        output = model(input_data)
        probabilities = F.softmax(output, dim=1)
        class_idx = torch.argmax(probabilities).item()

    # Map the index to class
    classes = {0: 'GALAXY', 1: 'STAR', 2: 'QSO'}
    return classes[class_idx]

# Get manual input
input_data = manual_input()

# Predict and display the class
predicted_class = predict_class(input_data, net, scaler)
print(f"The predicted class is: {predicted_class}")


# e.g. testing data

# 25.26307 22.66389 20.60976 20.25615 19.54544 1.424659
# 21.2611 20.50495 18.36379 23.17828 17.96264 0.2509563