import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import scipy.stats as stats

# Load the dataset
file_path = 'C:\Users\Abylaikhan Bari\Desktop\Data analysis\Datasets\HRDataset_v14.csv'
df = pd.read_csv(file_path)

# 1. Analyze and Clean the Dataset
# Convert date columns to datetime
date_columns = ['DateofHire', 'DateofTermination', 'LastPerformanceReview_Date']
df[date_columns] = df[date_columns].apply(pd.to_datetime, errors='coerce')

# Handle missing values in ManagerID
df['ManagerID'] = df['ManagerID'].fillna(-1)  # Or any other appropriate method

# 2. Exploratory Data Analysis (EDA)
print(df.describe())

# 3. Data Visualization
# Salary Distribution by Gender
plt.figure(figsize=(10, 6))
sns.boxplot(x='GenderID', y='Salary', data=df)
plt.title('Salary Distribution by Gender')
plt.xlabel('Gender (0 = Female, 1 = Male)')
plt.ylabel('Salary')
plt.show()

# Employee Satisfaction by Department
plt.figure(figsize=(12, 6))
sns.barplot(x='EmpSatisfaction', y='Department', data=df, estimator=lambda x: sum(x==5)/len(x) * 100)
plt.title('Percentage of Highly Satisfied Employees by Department')
plt.xlabel('Percentage of Employees with Highest Satisfaction (5)')
plt.ylabel('Department')
plt.show()

# Correlation Heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap of Numerical Variables')
plt.show()

# 4. Hypothesis Testing
# "Higher employee engagement is associated with higher performance scores."
performance_engagement = df.groupby('PerformanceScore')['EngagementSurvey'].mean().reset_index()
plt.figure(figsize=(10, 6))
sns.barplot(x='PerformanceScore', y='EngagementSurvey', data=performance_engagement)
plt.title('Average Engagement Survey Score by Performance Score')
plt.xlabel('Performance Score')
plt.ylabel('Average Engagement Survey Score')
plt.show()

anova_result = stats.f_oneway(df[df['PerformanceScore'] == 'Exceeds']['EngagementSurvey'],
                              df[df['PerformanceScore'] == 'Fully Meets']['EngagementSurvey'],
                              df[df['PerformanceScore'] == 'Needs Improvement']['EngagementSurvey'],
                              df[df['PerformanceScore'] == 'PIP']['EngagementSurvey'])
print('ANOVA result:', anova_result)

# 5. Regression Analysis
# Preparing data for regression
features = ['DeptID', 'PerfScoreID', 'PositionID', 'EmpSatisfaction', 'EngagementSurvey', 'GenderID', 'MaritalStatusID']
target = 'Salary'
X = df[features]
y = df[target]

# Encoding categorical variables
categorical_features = ['DeptID', 'PerfScoreID', 'PositionID', 'GenderID', 'MaritalStatusID']
one_hot = OneHotEncoder()
transformer = ColumnTransformer([("one_hot", one_hot, categorical_features)], remainder='passthrough')
X_encoded = transformer.fit_transform(X)

# Splitting the data
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Regression model
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluating the model
r2 = r2_score(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
print('R-squared:', r2)
print('RMSE:', rmse)
