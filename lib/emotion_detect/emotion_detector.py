import numpy as np
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
MODEL_PATH = os.path.dirname(os.path.abspath(__file__)) + '/model.h5'
XML_PATH = os.path.dirname(os.path.abspath(__file__)) + '/haarcascade_frontalface_default.xml'

emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}


class EmotionDetector:
    def __init__(self):
        self.model = Sequential()

        self.model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))
        self.model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))

        self.model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))

        self.model.add(Flatten())
        self.model.add(Dense(1024, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(7, activation='softmax'))

        self.model.load_weights(MODEL_PATH)

    def detect(self, frame):
        facecasc = cv2.CascadeClassifier(XML_PATH)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        try:
            faces = facecasc.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

            for (x, y, w, h) in faces:
                roi_gray = gray[y:y + h, x:x + w]
                cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
                prediction = self.model.predict(cropped_img)
                maxindex = int(np.argmax(prediction))
                emotion_result = emotion_dict[maxindex]

                return [x, y, emotion_result]
        except Exception as err:
            print(err)
            return None
