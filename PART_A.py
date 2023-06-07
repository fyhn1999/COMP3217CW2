import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score
import matplotlib.pyplot as plt

# Read the training data from 'TrainingDataBinary.csv' without header
data = pd.read_csv('TrainingDataBinary.csv', header=None)

# Extract the features (X) and the target variable (y)
X = data.drop(data.columns[-1], axis=1)
y = data[data.columns[-1]]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=39)

# Create and train the logistic regression model
logistic = LogisticRegression(max_iter=2000)
logistic.fit(X_train, y_train)

# Evaluate the model's performance on the testing set
score = logistic.score(X_test, y_test)
print("LogisticRegression score: %f" % score)

# Make predictions on the testing set
predictions = logistic.predict(X_test)

# Calculate the F1 score
f1_score = f1_score(y_test, predictions, average='macro')
print("F1 score: %f" % f1_score)

# Read the testing data from 'TestingDataBinary.csv' without header
test_data = pd.read_csv('TestingDataBinary.csv', header=None)

# Make predictions on the testing data
test_predictions = logistic.predict(test_data)

# Create a DataFrame to store the test predictions
test_results = pd.DataFrame(test_predictions)

# Concatenate the test data with the test predictions
test_results = pd.concat([test_data, test_results], axis=1)

# Save the test results to 'TestingResultsBinary.csv' without header
test_results.to_csv('TestingResultsBinary.csv', index=False, header=False)

# Calculate and display the confusion matrix
cm = confusion_matrix(y_test, predictions, labels=logistic.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=logistic.classes_)
disp.plot()
plt.show()
