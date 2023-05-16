# Code Requirements

To run this code, you need the following:
1. Python: Make sure you have Python installed on your system. You can download it from the [official Python website:](https://www.python.org/downloads/)
2. Required Libraries: This code requires the following libraries to be installed:
    - pandas
    - scikit-learn (sklearn)
    - matplotlib
    - seaborn
You can install these libraries using pip, a package installer for Python. Run the following command to install the required libraries:
```pip install pandas scikit-learn matplotlib seaborn```
3. Dataset: The code assumes that you have a CSV file named "emails.csv" containing the email data. Make sure to place the file in the same directory as the Python script or specify the correct file path in the pd.read_csv() function call.

# Code Functionality
This code performs the following tasks:

1. Reads and explores the email data from a CSV file.
2. Splits the data into training and test sets.
3. Converts the email text into a matrix of token counts using the CountVectorizer class from scikit-learn.
4. Trains a Naive Bayes classifier on the training data.
5. Performs cross-validation to evaluate the classifier's performance.
6. Uses grid search to find the best hyperparameters for the classifier.
7. Trains a new Naive Bayes classifier with the best hyperparameters.
8. Predicts the class labels for the test set.
9. Evaluates the classifier's performance by calculating accuracy and creating a confusion matrix.
10. Visualizes the confusion matrix using a heatmap.
11. Extracts the top spam and ham features based on their log probabilities.
12. Displays the top 10 spam and ham features.
13. Visualizes the top spam and ham features using bar plots.
14. The code provides insights into the distribution of spam and ham emails, the accuracy of the classifier, the best hyperparameters found through grid search, and the most important features for classifying spam and ham emails.

To run the code, execute it in a Python environment or run each section separately in a Jupyter Notebook or Python IDE. Make sure to have the required libraries and the dataset in the appropriate location.