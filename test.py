import tensorflow as tf
from ultralytics import YOLO
from flet import app, Page, Text, Image, ElevatedButton, Row, Column, Container, FloatingActionButton, Icon, IconButton, TextField, AlertDialog, SnackBar, ElevatedButton, Image, Row, Column, Container, FloatingActionButton, Icon, IconButton, TextField, AlertDialog, SnackBar
import speech_recognition as sr
import pyaudio
import cv2

# Load YOLO v8 model
model = YOLO("yolov8n.pt")  # Replace with your desired YOLO v8 model

def detect_objects(image):
    results = model(image)
    return results

def main(page):
    
    
        
    page.title = "Object Detection App"
    page.padding = 20

    # Create a text field for displaying detected objects
    detected_objects_text = Text("Detected Objects:", size=16)

    # Create a container to hold the detected objects list
    detected_objects_container = Container()

    

    

    # Create a function to handle voice input
    def voice_input_changed(e):
        if voice_input_text.value == "hey drusthi open the app":
            start_detection()

    # Create a function to start object detection
    def start_detection(e=None):
        # Capture image from camera
        cap = cv2.VideoCapture(0)

        while True:
            ret, frame = cap.read()

            if not ret:
                break

            # Convert frame to a PIL Image for YOLO v8
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)

            results = detect_objects(image)

            # Clear previous detected objects
            detected_objects_container.controls.clear()

            # Display detected objects
            for i, obj in enumerate(results.xyxy[0]):
                detected_objects_container.controls.append(
                    Text(text=f"{i+1}. {obj[5]}", size=14)
                )

            # Display the frame with detected objects (optional)
            cv2.imshow("Object Detection", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
     # Create a button to start object detection
    start_detection_button = ElevatedButton(text="Start Detection", on_click=start_detection)
    
    # Create a text field for voice input
    voice_input_text = TextField(label="Voice Input", on_change=voice_input_changed)
    
    page.add(
        Column([
            detected_objects_text,
            detected_objects_container,
            start_detection_button,
            voice_input_text,
        ])
    )

    app(target=main)

if __name__ == "__main__":
    app(target=main)