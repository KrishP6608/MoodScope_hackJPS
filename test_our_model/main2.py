import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load the pre-trained emotion recognition model
emotion_model = load_model('C:/Users/kp127/OneDrive/Desktop/HackJPS/models/emotionModel.h5')

# Load the face detection cascade
import cv2

# Load the cascade classifier
cascade_path = 'C:/Python311/Lib/site-packages/cv2/data/haarcascade_frontalcatface.xml'
face_cascade = cv2.CascadeClassifier(cascade_path)


# Function to predict emotion from a face image
def predict_emotion(face_image):
    face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
    face_image = cv2.resize(face_image, (256, 256))
    face_image = face_image.astype('float32') / 255.0
    face_image = np.expand_dims(face_image, axis=0)
    face_image = np.expand_dims(face_image, axis=-1)
    emotion_prediction = emotion_model.predict(face_image)
    return emotion_prediction


# Load the screen recording video
video_path = 'test_video.mp4'  # Replace with your screen recording file path
video = cv2.VideoCapture(video_path)

while True:
    # Read a frame from the video
    ret, frame = video.read()

    # Detect faces in the frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Extract the face region from the frame
        face_image = gray[y:y + h, x:x + w]

        # Predict the emotion for the face
        emotion_prediction = predict_emotion(face_image)
        emotion_label = np.argmax(emotion_prediction)
        emotion = ['Happiness', 'Neutral', 'Sadness', 'Anger', 'Surprise', 'Disgust', 'Fear'][emotion_label]

        # Draw bounding box and emotion label on the frame
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # Display the resulting frame
    cv2.imshow('Virtual Classroom Emotion Analysis', frame)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all windows
video.release()
cv2.destroyAllWindows()
# from statistics import mode

# import cv2
# from keras.models import load_model
# import numpy as np

# from utils.datasets import get_labels
# from utils.inference import detect_faces
# from utils.inference import draw_text
# from utils.inference import draw_bounding_box
# from utils.inference import apply_offsets
# from utils.inference import load_detection_model
# from utils.preprocessor import preprocess_input

# # parameters for loading data and images
# detection_model_path = 'C:/Users/kp127/OneDrive/Desktop/HackJPS/haarcascade_frontalface_default.xml'
# emotion_model_path = 'C:/Users/kp127/OneDrive/Desktop/HackJPS/models/emotionModel.h5'
# emotion_labels = get_labels('fer2013')

# # hyper-parameters for bounding boxes shape
# frame_window = 10
# emotion_offsets = (20, 40)

# # loading models
# face_detection = load_detection_model(detection_model_path)
# emotion_classifier = load_model(emotion_model_path, compile=False)

# # getting input model shapes for inference
# emotion_target_size = emotion_classifier.input_shape[1:3]

# # starting lists for calculating modes
# emotion_window = []

# # starting video streaming
# cv2.namedWindow('window_frame')
# video_capture = cv2.VideoCapture('C:/Users/kp127/OneDrive/Desktop/HackJPS/test _video.mp4')
# while True:
#     bgr_image = video_capture.read()[1]
#     gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
#     rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
#     faces = detect_faces(face_detection, gray_image)

#     for face_coordinates in faces:

#         x1, x2, y1, y2 = apply_offsets(face_coordinates, emotion_offsets)
#         gray_face = gray_image[y1:y2, x1:x2]
#         try:
#             gray_face = cv2.resize(gray_face, (emotion_target_size))
#         except:
#             continue

#         gray_face = preprocess_input(gray_face, True)
#         gray_face = np.expand_dims(gray_face, 0)
#         gray_face = np.expand_dims(gray_face, -1)
#         emotion_prediction = emotion_classifier.predict(gray_face)
#         emotion_probability = np.max(emotion_prediction)
#         emotion_label_arg = np.argmax(emotion_prediction)
#         emotion_text = emotion_labels[emotion_label_arg]
#         emotion_window.append(emotion_text)

#         if len(emotion_window) > frame_window:
#             emotion_window.pop(0)
#         try:
#             emotion_mode = mode(emotion_window)
#         except:
#             continue

#         if emotion_text == 'angry':
#             color = emotion_probability * np.asarray((255, 0, 0))
#         elif emotion_text == 'sad':
#             color = emotion_probability * np.asarray((0, 0, 255))
#         elif emotion_text == 'happy':
#             color = emotion_probability * np.asarray((255, 255, 0))
#         elif emotion_text == 'surprise':
#             color = emotion_probability * np.asarray((0, 255, 255))
#         else:
#             color = emotion_probability * np.asarray((0, 255, 0))

#         color = color.astype(int)
#         color = color.tolist()

#         draw_bounding_box(face_coordinates, rgb_image, color)
#         draw_text(face_coordinates, rgb_image, emotion_mode,
#                   color, 0, -45, 1, 1)

#     bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
#     cv2.imshow('window_frame', bgr_image)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
# video_capture.release()
# cv2.destroyAllWindows()

