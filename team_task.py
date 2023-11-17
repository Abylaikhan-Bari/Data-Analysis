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
df['DateofHire'] = pd.to_datetime(df['DateofHire'])
df['DateofTermination'] = pd.to_datetime(df['DateofTermination'])
df['LastPerformanceReview_Date'] = pd.to_datetime(df['LastPerformanceReview_Date'])

# Handle missing values
df = df.fillna(df.mean(numeric_only=True))

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

# 4. Optional: Hypothesis Testing
# "Higher engagement scores lead to higher employee satisfaction."
plt.figure(figsize=(8, 6))
sns.scatterplot(x='EngagementSurvey', y='EmpSatisfaction', data=df)
plt.title('Engagement Survey vs Employee Satisfaction')
plt.show()

# 5. Regression Analysis
# Predicting Salary based on EngagementSurvey and EmpSatisfaction
X = df[['EngagementSurvey', 'EmpSatisfaction']]
y = df['Salary']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R-squared:", r2_score(y_test, y_pred))

# Optional: Plotting predictions
plt.scatter(X_test['EngagementSurvey'], y_test, color='black', label='Actual Salary')
plt.scatter(X_test['EngagementSurvey'], y_pred, color='blue', label='Predicted Salary')
plt.xlabel('Engagement Survey')
plt.ylabel('Salary')
plt.legend()
plt.show()
