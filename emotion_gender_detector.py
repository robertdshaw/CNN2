from keras.models import load_model
from time import sleep
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
import cv2
import numpy as np

face_classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
emotion_classifier = load_model("best_emotion_model_30.keras")
gender_classifier = load_model("final_gender_model_pre_trained.keras")

emotion_labels = ['anger','disgust','fear','happy','neutral', 'sad', 'surprise']
gender_labels = ["male", "female"]

# Set the threshold for the gender model
threshold = 0.35  # Adjusted the threshold down from 0.50 to improve the recall for female class while maintaining reasonable balance with precision.

cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()
    #labels = []
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray)

    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)

        roi_gray = gray[y:y+h,x:x+w]
        roi_gray = cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)

        if np.sum([roi_gray]) != 0:
            roi = img_to_array(roi_gray.astype('float'))
            roi_gray = img_to_array(roi_gray)
            roi_gray = np.expand_dims(roi_gray, axis=0)

            roi_color = frame[y:y + h, x:x + w]
            roi_color = cv2.resize(roi_color, (224, 224), interpolation=cv2.INTER_AREA)
            roi_color = roi_color.astype('float32') / 127.5 - 1
            roi_color = img_to_array(roi_color)
            roi_color = np.expand_dims(roi_color, axis=0)
        
            prediction_emotion = emotion_classifier.predict(roi_gray)
            prediction_gender = gender_classifier.predict(roi_color)

            emotion_label = emotion_labels[np.argmax(prediction_emotion)]
            #gender_label = gender_labels[np.argmax(prediction_gender)]
            gender_label = gender_labels[int(prediction_gender > threshold)]

            emotion_label_position = (x,y-10)
            gender_label_position = (x,y+h+25)
       
            cv2.putText(frame, emotion_label, emotion_label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, gender_label, gender_label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(frame, 'No Faces', (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Emotion and Gender Detector', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
