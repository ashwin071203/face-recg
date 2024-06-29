import cv2
import numpy as np
import sqlite3
import streamlit as st
from PIL import Image

# Function to connect to the database
def connect_db():
    return sqlite3.connect('face_recognition.db')

# Function to load the database
def load_database():
    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute('SELECT face_id, name, contact, image FROM faces')
    data = cursor.fetchall()
    conn.close()
    return data

# Function to save a new face to the database
def save_to_database(name, contact, face_image):
    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute('INSERT INTO faces (name, contact, image) VALUES (?, ?, ?)', (name, contact, face_image))
    conn.commit()
    conn.close()

# Function to update a record in the database
def update_database(face_id, name, contact):
    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute('UPDATE faces SET name = ?, contact = ? WHERE face_id = ?', (name, contact, face_id))
    conn.commit()
    conn.close()

# Function to delete a record from the database
def delete_from_database(face_id):
    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute('DELETE FROM faces WHERE face_id = ?', (face_id,))
    conn.commit()
    conn.close()

# Function to convert image blob to image
def convert_blob_to_image(image_blob):
    image_np = np.frombuffer(image_blob, dtype=np.uint8)
    image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
    return image

# Function to detect faces
def detect_faces(frame):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    return faces

# Function to identify a face from the database
def identify_face(face, database):
    face_bytes = cv2.imencode('.png', face)[1].tobytes()
    for record in database:
        stored_face_id, name, contact, stored_face_bytes = record
        stored_face = convert_blob_to_image(stored_face_bytes)
        if stored_face is not None:
            stored_face_gray = cv2.cvtColor(stored_face, cv2.COLOR_BGR2GRAY)
            result = cv2.matchTemplate(cv2.cvtColor(face, cv2.COLOR_BGR2GRAY), stored_face_gray, cv2.TM_CCOEFF_NORMED)
            if result.max() > 0.7:  # Matching threshold
                return stored_face_id, {'name': name, 'contact': contact}
    return None, None

# Function to add a new face
def add_new_face(face):
    face_bytes = cv2.imencode('.png', face)[1].tobytes()
    return face_bytes

# Main function
def main():
    st.title('Facial Recognition App')

    choice = st.sidebar.selectbox('Choose Action', ['Home', 'Add New Face', 'View/Edit Records', 'Settings'])

    if choice == 'Home':
        st.write("Use the webcam to detect faces.")
        run = st.checkbox('Run')
        FRAME_WINDOW = st.image([])

        if run:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                st.error("Error: Could not access the webcam.")
                return

            new_face_detected = False
            database = load_database()
            face_image_bytes = None

            while run:
                ret, frame = cap.read()
                if not ret:
                    st.error("Error: Failed to capture image from webcam.")
                    break

                faces = detect_faces(frame)
                for (x, y, w, h) in faces:
                    face = frame[y:y+h, x:x+w]
                    face_id, details = identify_face(face, database)
                    if face_id:
                        text = f"Name: {details['name']} \nContact: {details['contact']}"
                    else:
                        if not new_face_detected:
                            face_image_bytes = add_new_face(face)
                            st.session_state['face_image_bytes'] = face_image_bytes
                            new_face_detected = True
                        text = "New face detected! Please go to 'Add New Face' to enter details."
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                    cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
                FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            cap.release()

    elif choice == 'Add New Face':
        st.write("Enter details for the new face:")
        new_name = st.text_input('Name')
        new_contact = st.text_input('Contact')
        
        if 'face_image_bytes' in st.session_state and st.session_state['face_image_bytes'] is not None:
            face_image = st.session_state['face_image_bytes']
            st.image(cv2.cvtColor(convert_blob_to_image(face_image), cv2.COLOR_BGR2RGB), caption='New Face Detected')

            if st.button('Save Details'):
                save_to_database(new_name, new_contact, face_image)
                st.session_state['face_image_bytes'] = None
                st.success("Details saved successfully!")
        else:
            st.write("No new face detected. Please go to 'Home' to detect a new face first.")

    elif choice == 'View/Edit Records':
        st.write("View and Edit Records")
        database = load_database()
        if st.checkbox('Refresh Database'):
            database = load_database()
        for record in database:
            st.write(f"ID: {record[0]}, Name: {record[1]}, Contact: {record[2]}")
            st.image(cv2.cvtColor(convert_blob_to_image(record[3]), cv2.COLOR_BGR2RGB), caption=f"Face ID: {record[0]}")

        record_to_edit = st.selectbox('Select Record to Edit', [record[0] for record in database])
        if record_to_edit:
            selected_record = [record for record in database if record[0] == record_to_edit][0]
            edit_name = st.text_input('Edit Name', selected_record[1])
            edit_contact = st.text_input('Edit Contact', selected_record[2])
            if st.button('Update'):
                update_database(record_to_edit, edit_name, edit_contact)
                st.success("Record updated successfully!")

        record_to_delete = st.selectbox('Select Record to Delete', [record[0] for record in database])
        if st.button('Delete'):
            delete_from_database(record_to_delete)
            st.success("Record deleted successfully!")

    elif choice == 'Settings':
        st.write("Settings")
        display_name = st.checkbox('Display Name', value=True)
        display_contact = st.checkbox('Display Contact', value=True)
        st.write("Settings saved successfully.")

if __name__ == "__main__":
    main()

