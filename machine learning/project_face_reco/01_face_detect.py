#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 16 11:17:47 2024

@author: minkhant
"""

import cv2 as cv 

cap = cv.VideoCapture(0)
path = '/Users/minkhant/MachineLearningI/machine learning/project_face_reco/haarcascade_frontalface_alt.xml'

face_cascade = cv.CascadeClassifier(path)

while True:
    ret, frame = cap.read()
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    
    if ret == False:
        continue 
    
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    for (x, y, w, h) in faces:
        cv.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv.rectangle(gray, (x, y), (x+w, y+h), (255, 54, 80), 2)
        
    cv.imshow('Frame', frame)
    # cv.imshow('Gray', gray)
    
    if cv.waitKey(1) == 27:
        break 

cap.release()
cv.destroyAllWindows()
    
    