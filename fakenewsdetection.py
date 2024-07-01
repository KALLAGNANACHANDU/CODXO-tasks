# Import necessary libraries
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the dataset (replace with your actual data)
data = pd.read_csv("/newss.csv")

# Handle missing values in the 'text' and 'label' columns
data.dropna(subset=['text', 'label'], inplace=True)  # Drop rows with missing 'text' or 'label' values

# Preprocess the text data
# (Add your preprocessing steps here)

# Extract features using TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data['text'])
y = data['label']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train a logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)  # Now this should work without the NaN error

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)