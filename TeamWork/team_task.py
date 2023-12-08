import psycopg2
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Database connection parameters
db_config = {
    "database": "postgres",
    "user": "postgres",
    "password": "root",
    "host": "127.0.0.1",
    "port": "5432"
}

# Connect to the database
conn = psycopg2.connect(**db_config)

# SQL query
query = "SELECT * FROM hrdataset_v14;"

# Fetch data into DataFrame
df = pd.read_sql(query, conn)

# Close the connection
conn.close()

# 1. Analyze and Clean the Dataset
# Convert date columns to datetime
date_columns = ['DateofHire', 'DateofTermination', 'LastPerformanceReview_Date']
df[date_columns] = df[date_columns].apply(pd.to_datetime, errors='coerce')

# Handle missing values
# Fill numeric columns with their mean and categorical with the mode
numeric_cols = df.select_dtypes(include=['number']).columns
df[numeric_cols] = df[numeric_cols].fillna(df.mean(numeric_only=True))
categorical_cols = df.select_dtypes(include=['object']).columns
df[categorical_cols] = df[categorical_cols].fillna(df[categorical_cols].mode().iloc[0])

# 2. Exploratory Data Analysis (EDA)
print(df.describe())

# 3. Data Visualization
# Salary Distribution
sns.histplot(df['Salary'])
plt.title('Salary Distribution')
plt.show()

# Relationship between EngagementSurvey, EmpSatisfaction, and PerformanceScore
sns.pairplot(df[['EngagementSurvey', 'EmpSatisfaction', 'PerformanceScore']])
plt.show()

# 4. Hypothesis Testing
# "Higher engagement scores lead to higher employee satisfaction."
plt.figure(figsize=(8, 6))
sns.scatterplot(x='EngagementSurvey', y='EmpSatisfaction', data=df)
plt.title('Engagement Survey vs Employee Satisfaction')
plt.show()

# 5. Regression Analysis
# Predicting Salary based on all other features
X = df_encoded.drop('Salary', axis=1)  # Drop the target variable to get the features
y = df_encoded['Salary']

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creating and fitting the model
model = LinearRegression()
model.fit(X_train, y_train)

# Making predictions on the entire dataset for correlation matrix
df_encoded['Predicted Salary'] = model.predict(df_encoded.drop('Salary', axis=1))

# Calculate the correlation matrix including both actual and predicted values
correlation_matrix = df_encoded.corr()

# Plotting the heatmap for the correlation matrix
plt.figure(figsize=(20, 18))  # Adjust the size of the heatmap to accommodate the larger number of variables
sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', center=0)
plt.title('Heatmap of Correlation Matrix including Predicted Salary')
plt.show()