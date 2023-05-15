from sklearn.ensemble import RandomForestClassifier
import pickle

# Load the training data
with open('training_data.pkl', 'rb') as f:
    training_data = pickle.load(f)

# Separate features and labels
features = [data[0] for data in training_data]
labels = [data[1] for data in training_data]

# Initialize a model
model = RandomForestClassifier(n_estimators=100)

# Train the model
model.fit(features, labels)

# Save the model
with open('coin_classifier.pkl', 'wb') as f:
    pickle.dump(model, f)