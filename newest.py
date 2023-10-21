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

# Execute a SELECT query to find the average number of rooms
cur.execute("SELECT rooms, AVG(rooms) as avg_rooms FROM newhouse GROUP BY rooms ORDER BY rooms;")

# Fetch and process the results
avg_rooms_rows = cur.fetchall()
print("\nAverage Number of Rooms:")
for row in avg_rooms_rows:
    print(row)

# Execute a SELECT query to find the average price grouped by the number of rooms
cur.execute("SELECT rooms, AVG(price) as avg_price FROM newhouse GROUP BY rooms ORDER BY rooms;")

# Fetch and process the results
avg_price_rows = cur.fetchall()
print("\nAverage Price of Apartments:")
for row in avg_price_rows:
    print(row)

# Close cursor and connection
cur.close()
conn.close()
