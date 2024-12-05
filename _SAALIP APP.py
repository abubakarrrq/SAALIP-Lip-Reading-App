#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import dlib
import numpy as np
from tensorflow import keras
import tkinter as tk
from PIL import Image, ImageTk

# Load your trained model here
custom_optimizer = keras.optimizers.Adam(learning_rate=0.001)
model = keras.models.load_model('C:/Users/Grace/Downloads/lip/LipsNew.h5', compile=False)
model.compile(optimizer=custom_optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

def preprocess_frame(frame):
    lips_frame = cv2.resize(frame, (64, 224))
    lips_frame = lips_frame / 255.0  # Normalize pixel values
    return lips_frame

# Initialize webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

frames_to_capture = 30
captured_frames = []

# Initialize face and lip detector from dlib
detector = dlib.get_frontal_face_detector()
lip_predictor = dlib.shape_predictor("C:/Users/Grace/Downloads/lip/shape_predictor_68_face_landmarks.dat")  # Replace with actual path

# Create a Tkinter window
root = tk.Tk()
root.title("SAALIP")

# Add title and subtitle
title_label = tk.Label(root, text="SAALIP", font=("Helvetica", 24, "bold"), fg="blue")
title_label.pack(pady=(10, 5))

subtitle_label = tk.Label(root, text="1st Pakistani Lip Reading App", font=("Helvetica", 16), fg="green")
subtitle_label.pack(pady=(0, 20))

# Create a label for showing the video feed
video_label = tk.Label(root)
video_label.pack()

# Create a label for showing capture status
status_label = tk.Label(root, text="Status: Waiting to capture")
status_label.pack()

# Function to update the video feed in the GUI
def update_video():
    ret, frame = cap.read()
    if ret:
        # Detect faces
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray_frame)
        
        # Draw green rectangle around lips
        for face in faces:
            shape = lip_predictor(gray_frame, face)
            lips_points = np.array([(shape.part(i).x, shape.part(i).y) for i in range(48, 61)])
            cv2.polylines(frame, [lips_points], True, (0, 255, 0), 2)
        
        # Display the frame
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame)
        frame = ImageTk.PhotoImage(image=frame)
        
        video_label.imgtk = frame
        video_label.configure(image=frame)
        
        video_label.after(30, update_video)
    else:
        print("Error: Could not read frame from webcam.")

# Start updating the video feed
update_video()

# Function to capture the lip-reading sequence and display the prediction
def capture_and_predict():
    global captured_frames
    global frames_to_capture
    
    # Disable the capture button to prevent re-capturing
    capture_button.config(state=tk.DISABLED)
    status_label.config(text="Status: Capturing frames...")
    
    for _ in range(frames_to_capture):
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame from webcam.")
            return
        
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        faces = detector(gray_frame)
        if len(faces) == 0:
            print("No faces detected.")
            continue
        
        for face in faces:
            shape = lip_predictor(gray_frame, face)
            lips_points = [shape.part(i) for i in range(48, 61)]
            
            min_x = max(min(point.x for point in lips_points) - 5, 0)
            max_x = min(max(point.x for point in lips_points) + 5, frame.shape[1])
            min_y = max(min(point.y for point in lips_points) - 5, 0)
            max_y = min(max(point.y for point in lips_points) + 5, frame.shape[0])
            
            lips = frame[min_y:max_y, min_x:max_x]
            
            if lips.shape[0] > 0 and lips.shape[1] > 0:
                captured_frames.append(lips)
                print(f"Captured frames: {len(captured_frames)}")
    
    # Predict only if enough frames are captured
    if len(captured_frames) >= frames_to_capture:
        sequence_length = 30
        sequence_frames = captured_frames[-sequence_length:]
        
        processed_sequence = [preprocess_frame(frame) for frame in sequence_frames]
        
        input_sequence = np.stack(processed_sequence, axis=0)
        
        predictions = model.predict(np.expand_dims(input_sequence, axis=0))
        
        predicted_class_index = np.argmax(predictions)
        labels = ['0','1','2','3','4','9','6','7','8','5','10']
        predicted_label = labels[predicted_class_index]

        prediction_text = f"Predicted: {predicted_label}"

        # Display the final prediction result on the same screen in red color
        prediction_label = tk.Label(root, text=prediction_text, font=("Helvetica", 16), fg="red")
        prediction_label.pack()

        # Re-enable the capture button and reset status
        capture_button.config(state=tk.NORMAL)
        status_label.config(text="Status: Waiting to capture")
        
        # Display the captured frames in a pop-up window
        cv2.imshow("Captured Frames", np.hstack(captured_frames))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        captured_frames = []
        frames_to_capture = 30
    else:
        print("Not enough frames captured for prediction.")
    
    # Re-enable the capture button and reset status
    capture_button.config(state=tk.NORMAL)
    status_label.config(text="Status: Waiting to capture")

# Button to trigger lip-reading and prediction
capture_button = tk.Button(root, text="Capture and Predict", command=capture_and_predict)
capture_button.pack()

# Function to quit the application
def quit_app():
    cap.release()
    root.destroy()

# Button to quit the application
quit_button = tk.Button(root, text="Quit", command=quit_app)
quit_button.pack()

# Start the Tkinter main loop
root.mainloop()

# Release the webcam and close the OpenCV windows after exiting the Tkinter loop
cap.release()
cv2.destroyAllWindows()


# In[ ]:




