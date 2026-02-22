import mysql.connector

# Connect to MySQL
conn = mysql.connector.connect(
    host="localhost",
    user="myuser",
    password="Seven_7777777_:0",
    database="mydatabase"
)

cursor = conn.cursor()

# Create table
cursor.execute("""
CREATE TABLE IF NOT EXISTS users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(100),
    age INT
)
""")

# Insert data
cursor.execute("INSERT INTO users (name, age) VALUES (%s, %s)", ("Martin", 30))
conn.commit()

# Fetch data
cursor.execute("SELECT * FROM users")
for row in cursor.fetchall():
    print(row)

cursor.close()
conn.close()
