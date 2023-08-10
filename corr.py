import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the data from the Excel file
data = pd.read_excel("updated_data.xlsx")

# Calculate the correlation matrix
correlation_matrix = data[[
    "Days Since Last Replacement",
    "Days Since Last Repair",
    "Consecutive Days of Same Task",
    "Frequency of Task Performed"
]].corr()

# Create a heatmap of the correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", center=0)
plt.title("Correlation Matrix")
plt.tight_layout()
plt.show()


