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

# Get the indices of the spam messages
spam_indices = data[data['class'] == 'spam'].index

# Get the messages from the spam messages
spam_messages = data.loc[spam_indices, 'message']

# Extract features from the spam messages
vectorizer = CountVectorizer(stop_words='english')
spam_features = vectorizer.fit_transform(spam_messages)

# Get the most common words in spam messages
word_counts = spam_features.sum(axis=0)
word_counts_df = pd.DataFrame(word_counts, columns=vectorizer.get_feature_names())
word_counts_df = word_counts_df.T
word_counts_df.columns = ['count']
word_counts_df = word_counts_df.sort_values('count', ascending=False).head(10)

# Plot the most common words in spam messages
plt.bar(word_counts_df.index, word_counts_df['count'])
plt.title('Most Common Words in Spam Messages')
plt.xlabel('Word')
plt.ylabel('Count')
plt.show()

# Get the most common sentences in spam messages
sentence_counts = pd.Series(spam_messages).value_counts().head(10)

# Plot the most common sentences in spam messages
plt.bar(sentence_counts.index, sentence_counts.values)
plt.title('Most Common Sentences in Spam Messages')
plt.xticks(rotation=90)
plt.xlabel('Sentence')
plt.ylabel('Count')
plt.show()

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
