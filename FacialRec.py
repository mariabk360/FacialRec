import cv2
import streamlit as st

face_cascade = cv2.CascadeClassifier('open.xml')

def detect_faces(min_neighbors, scale_factor, rectangle_color):
    # Initialize the webcam
    cap = cv2.VideoCapture(0)
    while True:
        # Read the frames from the webcam
        ret, frame = cap.read()
        # Convert the frames to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Detect the faces using the face cascade classifier
        faces = face_cascade.detectMultiScale(gray, scaleFactor=scale_factor, minNeighbors=min_neighbors)
        # Draw rectangles around the detected faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), rectangle_color, 2)
        # Display the frames
        cv2.imshow('Face Detection using Viola-Jones Algorithm', frame)
        # Exit the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # Release the webcam and close all windows
    cap.release()
    cv2.destroyAllWindows()

def app():
    st.title("Face Detection using Viola-Jones Algorithm")
    st.write("Press the button below to start detecting faces from your webcam")
    # Add instructions
    st.write("Instructions:")
    st.write("1. Click the 'Detect Faces' button to start detecting faces.")
    st.write("2. Adjust the 'minNeighbors' and 'scaleFactor' sliders to change the face detection parameters.")
    st.write("3. Use the color picker to choose the color of the rectangles around the detected faces.")
    st.write("4. Press 'q' to stop the face detection process.")

    # Add a button to start detecting faces
    if st.button("Detect Faces"):
        # Add sliders for minNeighbors and scaleFactor
        min_neighbors = st.slider("minNeighbors", 1, 10, 5)
        scale_factor = st.slider("scaleFactor", 1.1, 2.0, 1.3)

        # Add a color picker for rectangle color
        rectangle_color = st.color_picker("Rectangle Color", "#00FF00")

        # Convert the rectangle color to the correct format
        rectangle_color = tuple(int(rectangle_color[i:i+2], 16) for i in (1, 3, 5))

        # Call the detect_faces function with the chosen parameters
        detect_faces(min_neighbors, scale_factor, rectangle_color)

if __name__ == "__main__":
    app()
