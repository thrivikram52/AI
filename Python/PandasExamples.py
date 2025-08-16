import pandas as pd
import numpy as np
# 1. Creating a Series
# A Pandas Series is a one-dimensional labeled array capable of holding any data type.
series = pd.Series([1, 3, 5, 7, 9])
print("Series:\n", series)

# 2. Creating a DataFrame
# A DataFrame is a two-dimensional, size-mutable, and potentially heterogeneous tabular data.
data = {'Name': ['Alice', 'Bob', 'Charlie'], 'Age': [25, 30, 35], 'City': ['New York', 'San Francisco', 'Los Angeles']}
df = pd.DataFrame(data)
print("\nDataFrame:\n", df)


# Creating a DataFrame with columns
data = [
    [5.1, 3.5, 1.4, 0.2],
    [4.9, 3.0, 1.4, 0.2],
    [4.7, 3.2, 1.3, 0.2],
    [4.6, 3.1, 1.5, 0.2],
    [5.0, 3.6, 1.4, 0.2]
]
# Create the DataFrame
df = pd.DataFrame(data, columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])

# Display the DataFrame
print(df)


# 3. Reading Data from a CSV file
# Uncomment the following line to read from a CSV file (ensure you have 'data.csv' in the working directory).
# df = pd.read_csv('data.csv')

# 4. Viewing Data
print("\nFirst 2 Rows:\n", df.head(2))
print("\nLast 2 Rows:\n", df.tail(2))
print("\nDataFrame Shape:\n", df.shape)
print("\nSummary Statistics:\n", df.describe())

# 5. Selection and Indexing
print("\nSelecting 'Name' column:\n", df['Name'])
print("\nSelecting 'Name' and 'City' columns:\n", df[['Name', 'City']])
print("\nSelecting first row by index:\n", df.iloc[0])
print("\nSelecting first row by label:\n", df.loc[0])
print("\nSelecting rows where 'Age' > 30:\n", df[df['Age'] > 30])

# 6. Operations on DataFrames
df['Salary'] = [70000, 80000, 90000]  # Adding a new column
print("\nDataFrame with 'Salary' column added:\n", df)

df = df.drop('City', axis=1)  # Removing the 'City' column, axis=0 used to remove rows and axis=1 used to remove columns
print("\nDataFrame with 'City' column dropped:\n", df)

df = df.rename(columns={'Name': 'Employee Name'})  # Renaming columns
print("\nDataFrame with renamed columns:\n", df)

df['Age'] = df['Age'].apply(lambda x: x + 1)  # Applying functions
print("\nDataFrame with 'Age' incremented by 1:\n", df)

# 7. Handling Missing Data
df.loc[1, 'Age'] = np.nan  # Introducing a NaN value for demonstration
print("\nDataFrame with missing data:\n", df)

print("\nChecking for missing data:\n", df.isnull().sum())
df['Age'] = df['Age'].fillna(df['Age'].mean())  # Filling missing data
print("\nDataFrame with missing data filled:\n", df)

# 8. Grouping and Aggregation
grouped = df.groupby('Age')
print("\nGrouping by 'Age' and calculating mean 'Salary':\n", grouped['Salary'].mean())

# 9. Merging DataFrames
df1 = pd.DataFrame({'Name': ['Alice', 'Bob', 'Charlie'], 'Age': [25, 30, 35]})
df2 = pd.DataFrame({'Name': ['Alice', 'Bob', 'David'], 'Salary': [70000, 80000, 90000]})
merged_df = pd.merge(df1, df2, on='Name', how='inner')
print("\nMerged DataFrame:\n", merged_df)

# 10. Joining DataFrames
df1 = df1.set_index('Name')
df2 = df2.set_index('Name')
joined_df = df1.join(df2, how='inner')
print("\nJoined DataFrame:\n", joined_df)

# 11. Saving Data
# Saving DataFrame to a CSV file
df.to_csv('output.csv', index=False)
print("\nDataFrame saved to 'output.csv'.")

# Saving DataFrame to a JSON file
df.to_json('output.json')
print("\nDataFrame saved to 'output.json'.")
