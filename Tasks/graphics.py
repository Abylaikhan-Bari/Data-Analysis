import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine

# Define your database connection parameters
db_config = {
    "database": "postgres",
    "user": "postgres",
    "password": "root",
    "port": "5432",
    "host": "127.0.0.1"
}

# SQLAlchemy engine for Pandas
engine = create_engine(f'postgresql://{db_config["user"]}:{db_config["password"]}@{db_config["host"]}:{db_config["port"]}/{db_config["database"]}')

# SQL query to select the top 10 balances by region from the 'customers' table
# Adjust the column names to match the case used in your database
query = """
SELECT "Region", "Balance"
FROM customers
ORDER BY "Balance" DESC
LIMIT 10;
"""

# Execute the query and store the result in a pandas DataFrame
df = pd.read_sql(query, engine)

# Visualization with Seaborn
plt.figure(figsize=(10, 8))
sns.barplot(x="Balance", y="Region", data=df.sort_values("Balance", ascending=False))
plt.title('Top 10 Balances by Region')
plt.xlabel('Balance')
plt.ylabel('Region')
plt.tight_layout()
plt.show()
