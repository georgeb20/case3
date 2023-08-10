import pandas as pd
import matplotlib.pyplot as plt

# Load the data from the Excel file
data = pd.read_excel("data.xlsx")

# Convert the "Date" column to datetime type
data["Date"] = pd.to_datetime(data["Date"], dayfirst=True)  # Assuming date format is day-month-year

# Create a figure and axis
fig, ax = plt.subplots(figsize=(10, 6))

# Plotting the data
ax.plot(data["Date"], data["Task Performed"], marker='o', linestyle='-', color='b')

# Formatting
ax.set_title("Task Performances Over Time")
ax.set_xlabel("Date")
ax.set_ylabel("Task Performed")
ax.grid(True)

# Rotate x-axis labels for better visibility
plt.xticks(rotation=45)

# Show the plot
plt.tight_layout()
plt.show()
