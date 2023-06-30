import cv2
import streamlit as st
import numpy as np


def detect_faces(image, min_neighbors, scale_factor, rectangle_color):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=scale_factor, minNeighbors=min_neighbors
    )

    for x, y, w, h in faces:
        # Extract individual color channel values from rectangle_color
        rectangle_color_channels = [
            int(rectangle_color[i : i + 2], 16) for i in (1, 3, 5)
        ]
        rectangle_color_channels.reverse()  # Reverse the order of channels (BGR instead of RGB)
        rectangle_color_tuple = tuple(rectangle_color_channels)
        cv2.rectangle(image, (x, y), (x + w, y + h), rectangle_color_tuple, 2)

    return image


def main():
    st.write("## Face Detection App")
    st.write("Upload an image and detect faces using the Viola-Jones algorithm.")
    st.write("Follow the steps below:")

    st.write("### Instructions")
    st.write("1. Upload an image using the file uploader below.")
    st.write("2. Adjust the parameters for face detection.")
    st.write(
        "3. Use the color picker to choose the color of the rectangles drawn around the detected faces."
    )
    st.write("4. The image with detected faces will be displayed below.")
    st.write(
        "5. Click the 'Save Image' button to save the image with detected faces to your device."
    )

    # Upload image
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

    # Face detection parameters
    min_neighbors = st.slider("minNeighbors", min_value=1, max_value=10, value=3)
    scale_factor = st.slider(
        "scaleFactor", min_value=1.1, max_value=2.0, value=1.3, step=0.1
    )
    rectangle_color = st.color_picker("Rectangle Color", "#FF0000")

    if uploaded_file is not None:
        # Read the image
        image = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), 1)

        # Perform face detection
        image_with_faces = detect_faces(
            image, min_neighbors, scale_factor, rectangle_color
        )

        # Display the image with detected faces
        st.image(image_with_faces, channels="BGR")

        # Save the image with detected faces
        if st.button("Save Image"):
            cv2.imwrite("image_with_faces.jpg", image_with_faces)
            st.write("Image saved successfully!")


if __name__ == "__main__":
    main()
