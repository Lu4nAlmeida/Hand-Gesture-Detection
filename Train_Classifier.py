import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the array files from "Create_Dataset.py"
data_dict = pickle.load(open('./data.pickle', 'rb'))

# Save the arrays into their respective variable
data = np.asarray(data_dict['data'])  # They are actually saved as lists so that's why we convert to array
labels = np.asarray(data_dict['labels'])

# Split data into two sets one for training and one for testing
# Test set only contains 20% of the data, the other 80% is used for training
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# Define classifier
model = RandomForestClassifier()

# Train classifier
model.fit(x_train, y_train)

# Test classifier
y_predict = model.predict(x_test)

# Compute accuracy
score = accuracy_score(y_predict, y_test)
print('{}% of samples were classified correctly'.format(score * 100))

# Save our trained classifier as model.p
f = open('model.p', 'wb')
pickle.dump({'model': model}, f)
f.close()
