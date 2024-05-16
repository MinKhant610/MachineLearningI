#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 16 12:05:36 2024

@author: minkhant
"""

import numpy as np
import cv2 as cv
import os

cap = cv.VideoCapture(0)
path = '/Users/minkhant/MachineLearningI/machine learning/project_face_reco/haarcascade_frontalface_alt.xml'
face_cascade = cv.CascadeClassifier(path)
data_path = '/Users/minkhant/MachineLearningI/machine learning/project_face_reco/data/'

face_data = []
labels = []
name = {}
class_id = 0

def distance(x, X):
    return np.sqrt(np.sum((x - X) ** 2))

def knn(X, Y, x, K=5):
    m = X.shape[0]
    x = x.flatten()
    val = []
    for i in range(m):
        xi = X[i].flatten()
        dist = distance(x, xi)
        val.append((dist, Y[i][0]))  # Ensure labels are scalars
    
    val = sorted(val, key=lambda x: x[0])[:K]
    
    # print the val array to check its structure
    print("val array:", val)
    
    val = np.asarray(val)
    new_vals = np.unique(val[:, 1], return_counts=True)
    index = new_vals[1].argmax()
    output = new_vals[0][index]

    return output

for file in os.listdir(data_path):
    if file.endswith(".npy"):
        data_item = np.load(os.path.join(data_path, file))
        face_data.append(data_item)
        name[class_id] = file[:-4]
        target = class_id * np.ones((data_item.shape[0],), dtype=int)
        class_id += 1
        labels.append(target)

face_data_set = np.concatenate(face_data, axis=0)
face_labels = np.concatenate(labels, axis=0).reshape(-1, 1)

# print the shape of the dataset
print("Face data set shape:", face_data_set.shape)
print("Face labels shape:", face_labels.shape)

while True:
    ret, frame = cap.read()

    if not ret:
        continue

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    faces = sorted(faces, key=lambda f: f[2] * f[3])

    for face in faces[-1:]:
        x, y, w, h = face
        face_section = gray[y:y + h, x:x + w]
        face_section = cv.resize(face_section, (100, 100))

        predict = knn(face_data_set, face_labels, face_section)
        predict_name = name[int(predict)]
        cv.putText(frame, predict_name, (x, y - 30), cv.FONT_HERSHEY_SIMPLEX,
                   1, (255, 0, 0), 2, cv.LINE_AA)
        cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0))

    cv.imshow('Camera', frame)

    key = cv.waitKey(1)

    if key == 27:
        break

cap.release()
cv.destroyAllWindows()
