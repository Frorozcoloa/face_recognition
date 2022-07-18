import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

from pyrsistent import m

path = "images"
images = []
className = []
myList = os.listdir(path)

# Leemos las imagenes de la carpeta
for cl in myList:
    curImg = cv2.imread(f"{path}/{cl}")
    images.append(curImg)
    className.append(os.path.splitext(cl)[0])



def findEncoding(images:list)->list:
    """Coge las imagenes de la base de datos y le saca el encoding

    Args:
        images (list): una lista de arays con las imagenes

    Returns:
        list: encode de las imagenes.
    """    

    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

def markAttandace(name):
    """creating register"""
    with open("attendece.csv","r+") as f:
        myDataList = f.readline()
        nameList = []
        for line in myDataList:
            entry = line.split(",")
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dString = now.strftime("%H:%M:%S")
            f.writelines(f"\n{name},{dString}")





encodeListKnow = findEncoding(images)
print("Encoding Complete")

cap = cv2.VideoCapture(0)

while True:
    succes, img = cap.read()
    imgs = cv2.resize(img, (0,0), None, 0.25, 0.25)
    imgs = cv2.cvtColor(imgs, cv2.COLOR_BGR2RGB)

    # Primero reconocemos las imagenes y luego sacamos el encode
    faceCurFrame = face_recognition.face_locations(imgs)
    encodeCurFrame = face_recognition.face_encodings(imgs, faceCurFrame)

    # Buscamos por cada cara cual es el menor de la lista de encodes
    for encodeFace, faceLoc in zip(encodeCurFrame, faceCurFrame):
        matches = face_recognition.compare_faces(encodeListKnow,encodeFace)
        faceDist = face_recognition.face_distance(encodeListKnow,encodeFace)
        matchIndex = np.argmin(faceDist)

        #Buscamos el match y lo ponemos en pantalla
        if matches[matchIndex]:
            name = className[matchIndex].upper()
            y1,x2,y2,x1 = faceLoc
            y1,x2,y2,x1 = y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
            markAttandace(name)
    cv2.imshow("webcam", img)
    cv2.waitKey(1)


