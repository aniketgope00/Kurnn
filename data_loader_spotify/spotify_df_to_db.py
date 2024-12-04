import sqlite3
import pandas

def display_db_head(db_file, table_name):
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    query = f"SELECT * FROM {table_name} LIMIT 5"
    cursor.execute(query)
    rows = cursor.fetchall()
    for row in rows:
        print(row)

    conn.close()

spotify_df = pandas.read_csv("spotify_data.csv")
print("DataFrame loaded successfully.")
print(f'DataFrame Shape: {spotify_df.shape}')

#connecting to spotify_db
connection = sqlite3.connect('spotify_db.sqlite3')
spotify_df.to_sql('Data', connection, if_exists='replace', index=True)

db_file = "spotify_db.sqlite3"
table_name = "Data"
display_db_head(db_file, table_name)

connection.close()