import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Load the data
df = pd.read_csv('star_classification.csv')

# Separate features and target for classification
X = df.drop(['class', 'redshift'], axis=1)
y_class = df['class']

# Encode the class labels
encoder = OneHotEncoder(sparse_output=False)
y_class_encoded = encoder.fit_transform(y_class.values.reshape(-1, 1))

# Normalize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the dataset into training and testing sets
X_train, X_test, y_train_class, y_test_class = train_test_split(
    X_scaled, y_class_encoded, test_size=0.2, random_state=42)

# Define the classification model
model_class = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(y_train_class.shape[1], activation='softmax')
])

# Compile the model
model_class.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model_class.fit(X_train, y_train_class, epochs=50, batch_size=32, validation_split=0.1)

# Evaluate the model
class_loss, class_accuracy = model_class.evaluate(X_test, y_test_class)
print(f"Classification Model - Loss: {class_loss}, Accuracy: {class_accuracy}")