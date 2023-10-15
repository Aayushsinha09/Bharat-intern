import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the Titanic dataset (You can replace this with your own data)
titanic_data = pd.read_csv('titanic.csv')

# Select relevant features (socio-economic status, age, gender)
features = titanic_data[['Pclass', 'Age', 'Sex']]
labels = titanic_data['Survived']

# Handle missing values (e.g., filling missing ages with the mean age)
features['Age'].fillna(features['Age'].mean(), inplace=True)

# Convert categorical variables (e.g., gender) to numerical
features['Sex'] = features['Sex'].map({'female': 0, 'male': 1})

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Create and train a Decision Tree classifier
classifier = DecisionTreeClassifier(random_state=42)
classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = classifier.predict(X_test)

# Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Generate a classification report for more detailed metrics
report = classification_report(y_test, y_pred)
print(report)
