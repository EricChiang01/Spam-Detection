import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('emails.csv')

# Explore and preprocess the data
print("Data Summary:")
print(df.head())
print("\nNumber of emails:", len(df))
print("Number of spam emails:", df['spam'].sum())
print("Number of ham emails:", len(df) - df['spam'].sum())

df['spam'].value_counts().plot(kind='bar', color=['salmon', 'lightblue'])
plt.title('Distribution of Spam and Ham Emails')
plt.xlabel('Class')
plt.ylabel('Count')
plt.show()

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['spam'], test_size=0.2, random_state=42)

# Convert emails to a matrix of token counts
cv = CountVectorizer(stop_words='english')
X_train_counts = cv.fit_transform(X_train)
X_test_counts = cv.transform(X_test)

# Train a Naive Bayes classifier
clf = MultinomialNB()

# Perform cross-validation
scores = cross_val_score(clf, X_train_counts, y_train, cv=5)
mean_accuracy = scores.mean()

# Print the mean accuracy across folds
print("Cross-Validation Accuracy: {:.2f}%".format(mean_accuracy * 100))

plt.plot(range(1, 6), scores*100, marker='o')
plt.title('Cross-Validation Scores')
plt.xlabel('Fold')
plt.ylabel('Accuracy (%)')
plt.show()

# Define the parameter grid
param_grid = {
    'alpha': [0.1, 0.01, 0.001],
    'fit_prior': [True, False]
}

# Perform grid search
grid_search = GridSearchCV(clf, param_grid, cv=5)
grid_search.fit(X_train_counts, y_train)

# Print the best hyperparameters and corresponding accuracy
best_params = grid_search.best_params_
best_accuracy = grid_search.best_score_
print("Best Hyperparameters:", best_params)
print("Best Accuracy: {:.2f}%".format(best_accuracy * 100))

# Train a Naive Bayes classifier with the best hyperparameters
clf = MultinomialNB(alpha=best_params['alpha'], fit_prior=best_params['fit_prior'])
clf.fit(X_train_counts, y_train)

# Predict on the test set
predictions = clf.predict(X_test_counts)

# Evaluate the model
accuracy = accuracy_score(y_test, predictions)
confusion_mat = confusion_matrix(y_test, predictions)

# Print the evaluation metrics
print("\nModel Evaluation:")
print("Accuracy: {:.2f}%".format(accuracy * 100))
print("Confusion Matrix:")
print(confusion_mat)

sns.heatmap(confusion_mat, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Extract feature importance
feature_names = cv.get_feature_names()
top_spam_features = sorted(zip(feature_names, clf.feature_log_prob_[1]), key=lambda x: x[1], reverse=True)[:10]
top_ham_features = sorted(zip(feature_names, clf.feature_log_prob_[0]), key=lambda x: x[1], reverse=True)[:10]

print("\nTop 10 Spam Features:")
for feature, log_prob in top_spam_features:
    print(feature, "->", log_prob)

print("\nTop 10 Ham Features:")
for feature, log_prob in top_ham_features:
    print(feature, "->", log_prob)

# Top Spam Features
features, log_probs = zip(*top_spam_features)
plt.barh(features, log_probs, color='salmon')
plt.title('Top 10 Spam Features')
plt.xlabel('Log Probability')
plt.ylabel('Feature')
plt.gca().invert_yaxis()
plt.show()

# Top Ham Features
features, log_probs = zip(*top_ham_features)
plt.barh(features, log_probs, color='lightblue')
plt.title('Top 10 Ham Features')
plt.xlabel('Log Probability')
plt.ylabel('Feature')
plt.gca().invert_yaxis()
plt.show()