import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Suppress all warnings within this context
with warnings.catch_warnings():
    warnings.simplefilter("ignore")

    # Your code that may produce warnings

# Warnings will be shown again outside the context

# Load the data from the Excel file
data = pd.read_excel("updated_data.xlsx")

# Convert the "Date" column to datetime type
data["Date"] = pd.to_datetime(data["Date"], dayfirst=True)

# Define the features (X) and target (y)
features = [
    "Days Since Last Replacement",
    "Days Since Last Repair",
    "Consecutive Days of Same Task",
    "Frequency of Task Performed"
]
target = "Task Performed"

X = data[features]
y = data[target]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train a Decision Tree Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)

print(f"Model Accuracy: {accuracy:.2f}")

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

# ... (Previous code to load data and train the model)

# Calculate additional classification metrics
precision = precision_score(y_test, y_pred, average="weighted")
recall = recall_score(y_test, y_pred, average="weighted")
f1 = f1_score(y_test, y_pred, average="weighted")

print(f"Model Precision: {precision:.2f}")
print(f"Model Recall: {recall:.2f}")
print(f"Model F1 Score: {f1:.2f}")

# Generate a classification report
# Generate a classification report
# Generate a classification report
# Generate a classification report
class_names = y.unique()
print(class_names)

classification_rep = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)

# Convert the classification report dictionary into a DataFrame
classification_df = pd.DataFrame(classification_rep).transpose()

# Plot precision, recall, and F1-score for each class
plt.figure(figsize=(10, 6))
# Extract precision, recall, and f1-score from the DataFrame
metrics_to_plot = classification_df.loc[class_names, ['precision', 'recall', 'f1-score']]
class_names = ['Replace', 'Repair', 'PM', 'Testing']
print(class_names)
metrics_to_plot.plot(kind='bar', colormap="viridis")
plt.title("Classification Metrics per Task Type")
plt.xlabel("Task Type")
plt.ylabel("Score")
plt.legend(["Precision", "Recall", "F1-Score"])
plt.tight_layout()
plt.xticks(range(len(class_names)), class_names, rotation=45)  # Set x-axis labels
plt.show()
# Generate a confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred, labels=class_names)  # Normalize the matrix
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt=".2f", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
# Predict task for a new date
new_date = pd.to_datetime("2023-08-01", dayfirst=True)
new_features = [40, 80, 90, 2]  # Replace these values with the actual feature values
predicted_task = model.predict([new_features])[0]

print(y_test)
print(y_pred)
