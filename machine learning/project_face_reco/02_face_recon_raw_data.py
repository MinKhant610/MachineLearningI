#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 16 11:31:56 2024

@author: minkhant
"""

import cv2 as cv
import numpy as np 

cap = cv.VideoCapture(0)
path = '/Users/minkhant/MachineLearningI/machine learning/project_face_reco/haarcascade_frontalface_alt.xml'
face_cascade = cv.CascadeClassifier(path)
face_section = np.zeros((100, 100), dtype='uint8')
face_data = []

name = input("Enter your name ")

skip = 0 


while True:
    ret, frame = cap.read()
    
    if ret == False:
        continue 
        
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    faces = sorted(faces, key=lambda f:f[2]*f[3])
    
    for face in faces[-1 : ]:
        x, y, w, h = face 
        face_section = gray[y : y+h , x : x+w]
        face_section = cv.resize(face_section, (100, 100))
        cv.putText(frame, name, (x, y-30), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv.LINE_AA)
        cv.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 0))
    
    cv.imshow('Camera ', frame)
    
    if skip%10 == 0:
        face_data.append(face_section)
    skip += 1 
    
    key = cv.waitKey(1)
    
    if key == 27:
        break
face_data = np.asarray(face_data)
face_data = face_data.reshape((face_data.shape[0], -1))
np.save('/Users/minkhant/MachineLearningI/machine learning/project_face_reco/data/' + name + '.npy', face_data)

cap.release()
cv.destroyAllWindows()