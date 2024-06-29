import sqlite3

def create_database():
    conn = sqlite3.connect('face_recognition.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS faces (
            face_id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            contact TEXT NOT NULL,
            image BLOB NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

if __name__ == "__main__":
    create_database()
    print("Database and table created successfully.")
