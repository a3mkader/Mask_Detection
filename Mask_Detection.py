import time
from keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import cv2 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import  Dropout, Flatten, Dense, Conv2D, MaxPooling2D
from importlib import reload


class MaskDetection:

    def __init__(self):
        self.with_mask = []
        self.without_mask = []
        self.y = []
        self.model_trained = False
        self.model = load_model("./mask_detection.h5")
        self.results = {0: 'without mask', 1: 'with mask'}
        self.color = {0: (0, 0, 255), 1: (0, 255, 0)}
        self.rect_size = 4
        self.haarcascade = cv2.CascadeClassifier(
            './haarcascade_frontalface_default.xml')


    def read_data(self):
        # Data path
        train_dir = './Dataset/train/'
        test_dir = './Dataset/test/'
        train_with_mask = os.path.join(train_dir, 'with_mask')
        train_without_mask = os.path.join(train_dir, 'without_mask')
        img_with_mask = os.listdir(train_with_mask)
        img_without_mask = os.listdir(train_without_mask)

        # Read Data
        for img in img_with_mask:
            image = cv2.imread(os.path.join(train_with_mask, img))
            image = cv2.resize(image, (120, 120))
            self.with_mask.append(image)
            self.y.append(1)
        for img in img_without_mask:
            image = cv2.imread(os.path.join(train_without_mask, img))
            image = cv2.resize(image, (120, 120))
            self.without_mask.append(image)
            self.y.append(0)

    def preprocessing_data(self):
        #Preprocessing Data
        self.y = np.array(self.y)
        self.with_mask = np.array(self.with_mask)
        self.with_mask = self.with_mask.reshape(658, 120, 120, 3)
        self.without_mask = np.array(self.without_mask)
        self.without_mask = self.without_mask.reshape(657, 120, 120, 3)
        self.x = np.concatenate((self.with_mask, self.without_mask), axis=0)

        # split data
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            self.x, self.y, test_size=0.2, random_state=42)

        # one_hot encoding
        self.y_train_scaled = to_categorical(self.y_train)
        self.y_test_scaled = to_categorical(self.y_test)

        # rescaling
        self.x_train = self.x_train/255
        self.x_test = self.x_test/255

    def create_model(self):
        #Modeling
        early_stop = EarlyStopping(monitor='val_loss',patience=2)
        self.model = Sequential()
        self.model.add(Conv2D(filters=24, kernel_size=(4, 4),
                activation='relu', input_shape=(120, 120, 3)))
        self.model.add(MaxPooling2D(pool_size=(4, 4)))

        self.model.add(Conv2D(filters=24, kernel_size=(4, 4),
                activation='relu', input_shape=(120, 120, 3)))
        self.model.add(MaxPooling2D(pool_size=(4, 4)))

        self.model.add(Flatten())
        self.model.add(Dense(128, activation='relu'))
        self.model.add(Dropout(0.5))

        self.model.add(Dense(2, activation='softmax'))
        self.model.compile(loss='categorical_crossentropy',
                    optimizer='adam', metrics=['accuracy'])
        self.model.summary()
        self.model.fit(self.x_train, self.y_train_scaled,
                   epochs=8, validation_data=(self.x_test, self.y_test_scaled),
                callbacks=[early_stop])
        self.model.save('mask_detection.h5')

    def evaluate_model(self):
        if(not self.model_trained):
            self.read_data()
            self.preprocessing_data()
        test_scores = self.model.evaluate(self.x_test, self.y_test_scaled, verbose=0)
        train_scores = self.model.evaluate(self.x_train, self.y_train_scaled, verbose=0)
        train_acc=train_scores[1]
        train_loss=train_scores[0]
        test_acc = test_scores[1]
        test_loss =test_scores[0]

        return  [train_acc,train_loss,test_acc,test_loss]

    def train(self):
        self.read_data()
        self.preprocessing_data()
        self.create_model()
        self.model_trained= True

    def predict_live(self):
        cap = cv2.VideoCapture(0)
        while (cap.isOpened()):
            rval, im= cap.read()
            if rval:
                im = cv2.flip(im, 1, 1)
                gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
                rerect_size = cv2.resize(
                    gray, (im.shape[1] // self.rect_size, im.shape[0] // self.rect_size))
                faces = self.haarcascade.detectMultiScale(rerect_size)
                for f in faces:
                    (x, y, w, h) = [v * self.rect_size for v in f]

                    face_img = im[y:y+h, x:x+w]
                    rerect_sized = cv2.resize(face_img, (120, 120))
                    normalized = rerect_sized/255.0
                    reshaped = np.reshape(normalized, (1, 120, 120, 3))
                    reshaped = np.vstack([reshaped])
                    result = self.model.predict(reshaped)

                    label = np.argmax(result, axis=1)[0]

                    cv2.rectangle(im, (x, y), (x+w, y+h),
                                self.color[label], 2)
                    cv2.rectangle(im, (x, y-40), (x+w, y),
                                self.color[label], -1)
                    cv2.putText(im, self.results[label], (x, y-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.imshow('LIVE',   im)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
            else:
                break
        cap.release()
        cv2.waitKey(0)

        cv2.destroyAllWindows()
        

        
    def predict_video(self,video):
        cap = cv2.VideoCapture(video)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        while (True):
            rval, im= cap.read()
           
            if rval:
               time.sleep(1/fps)
               if cv2.waitKey(1) & 0xFF ==27:
                   break
            else:
                break
            im = cv2.flip(im, 1, 1)
            gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

            rerect_size = cv2.resize(
                gray, (im.shape[1] // self.rect_size, im.shape[0] // self.rect_size))
            faces = self.haarcascade.detectMultiScale(rerect_size)
            for f in faces:
                (x, y, w, h) = [v * self.rect_size for v in f]

                face_img = im[y:y+h, x:x+w]
                rerect_sized = cv2.resize(face_img, (120, 120))
                normalized = rerect_sized/255.0
                reshaped = np.reshape(normalized, (1, 120, 120, 3))
                reshaped = np.vstack([reshaped])
                result = self.model.predict(reshaped)

                label = np.argmax(result, axis=1)[0]

                cv2.rectangle(im, (x, y), (x+w, y+h), self.color[label], 2)
                cv2.rectangle(im, (x, y-40), (x+w, y), self.color[label], -1)
                cv2.putText(im, self.results[label], (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.imshow('Video', im)   
         
        cap.release()
        cv2.waitKey(0)

        cv2.destroyAllWindows()
        


    def predict_img(self,img):

        img=cv2.imread(img)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.haarcascade.detectMultiScale(gray, 1.1 , 3)
        for (x, y, w, h) in faces:
            face_img = img[y:y+h, x:x+w]
            rerect_sized = cv2.resize(face_img, (120, 120))
            normalized = rerect_sized/255.0
            reshaped = np.reshape(normalized, (1, 120, 120, 3))
            reshaped = np.vstack([reshaped])
            result = self.model.predict(reshaped)

            label = np.argmax(result, axis=1)[0]

            cv2.rectangle(img, (x, y), (x+w, y+h), self.color[label], 2)
            cv2.rectangle(img, (x, y-40), (x+w, y), self.color[label], -1)


            cv2.putText(img, self.results[label], (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.imshow('img', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
