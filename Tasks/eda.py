import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine

# Replace the placeholders with your actual database connection details
db_url = 'postgresql://postgres:root@localhost:5432/postgres'
query = "SELECT * FROM mtcars"

# Establish a database connection
engine = create_engine(db_url)

# Execute the SQL query and load data into a DataFrame
try:
    df = pd.read_sql(query, con=engine)

    # Perform exploratory data analysis (EDA) here
    # For example, you can use df.head(), df.info(), and various visualizations

    # 3. Checking the types of data
    print(df.dtypes)

    # 4. Dropping irrelevant columns
    df = df.drop(columns=['cyl'])

    # 5. Renaming the columns
    df = df.rename(columns={'mpg': 'MilesPerGallon'})

    # 6. Dropping the duplicate rows
    df = df.drop_duplicates()

    # 7. Dropping the missing or null values
    df = df.dropna()

    # 8. Detecting Outliers
    sns.boxplot(x=df['MilesPerGallon'])
    plt.title('Box Plot of Miles Per Gallon (MPG)')
    plt.show()

    # 9. Plot different features against one another (scatter), against frequency (histogram)
    plt.scatter(df['MilesPerGallon'], df['hp'])
    plt.title('Scatter Plot: MPG vs. Horsepower')
    plt.xlabel('Miles Per Gallon (MPG)')
    plt.ylabel('Horsepower (hp)')
    plt.show()

    sns.histplot(df['MilesPerGallon'], bins=10, kde=True)
    plt.title('Histogram of Miles Per Gallon (MPG)')
    plt.xlabel('Miles Per Gallon (MPG)')
    plt.show()

finally:
    # Close the database connection
    engine.dispose()
