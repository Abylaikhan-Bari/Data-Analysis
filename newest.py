import psycopg2

# Connect to the database
conn = psycopg2.connect(
    database="postgres",
    user="postgres",
    password="root",
    port="5432",
    host="127.0.0.1"
)

# Create a cursor
cur = conn.cursor()

# Execute a SELECT query
cur.execute ("SELECT * FROM newhouse ORDER BY price DESC LIMIT 1;")

# Fetch and process the results
rows = cur.fetchall()
for row in rows:
    print(row)

# Close cursor and connection
cur.close()
conn.close()
