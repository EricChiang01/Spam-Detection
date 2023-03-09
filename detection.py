import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

# Load the data
data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip', sep='\t', header=None, names=['class', 'message'])

# Preprocess the input messages
data['message'] = data['message'].str.lower()
data['message'] = data['message'].str.replace('[^\w\s]', '')
data['message'] = data['message'].str.replace('\d+', '')

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data['message'], data['class'], test_size=0.2, random_state=42)

# Extract features from the messages
vectorizer = CountVectorizer(stop_words='english')
X_train_features = vectorizer.fit_transform(X_train)
X_test_features = vectorizer.transform(X_test)

# Train a Naive Bayes classifier
clf = MultinomialNB()
clf.fit(X_train_features, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test_features)

# Evaluate the performance of the classifier
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, pos_label='spam')
recall = recall_score(y_test, y_pred, pos_label='spam')
f1 = f1_score(y_test, y_pred, pos_label='spam')

print(f'Test Set Performance:')
print(f'Accuracy: {accuracy:.2f}')
print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f'F1-score: {f1:.2f}\n')

# Train a Naive Bayes classifier on the entire dataset
X_features = vectorizer.fit_transform(data['message'])
y = data['class']
clf = MultinomialNB()
clf.fit(X_features, y)

# Get input from the user and preprocess it
user_input = input("Enter a message: ")
user_input = user_input.lower()
user_input = user_input.replace('[^\w\s]', '')
user_input = user_input.replace('\d+', '')

# Extract features from the user input
user_input_features = vectorizer.transform([user_input])

# Make predictions on the user input
prediction = clf.predict(user_input_features)[0]

# Print the prediction
if prediction == 'spam':
    print("This message is spam.")
else:
    print("This message is not spam.")
