import cv2
import numpy as np
import face_recognition
import os

from pyrsistent import m

path = "images"
images = []
className = []
myList = os.listdir(path)

for cl in myList:
    curImg = cv2.imread(f"{path}/{cl}")
    images.append(curImg)
    className.append(os.path.splitext(cl)[0])



def findEncoding(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

encodeListKnow = findEncoding(images)
print("Encoding Complete")

cap = cv2.VideoCapture(0)

while True:
    succes, img = cap.read()
    imgs = cv2.resize(img, (0,0), None, 0.25, 0.25)
    imgs = cv2.cvtColor(imgs, cv2.COLOR_BGR2RGB)

    faceCurFrame = face_recognition.face_locations(imgs)
    encodeCurFrame = face_recognition.face_encodings(imgs, faceCurFrame)

    for encodeFace, faceLoc in zip(encodeCurFrame, faceCurFrame):
        matches = face_recognition.compare_faces(encodeListKnow,encodeFace)
        faceDist = face_recognition.face_distance(encodeListKnow,encodeFace)
        matchIndex = np.argmin(faceDist)

        if matches[matchIndex]:
            name = className[matchIndex].upper()
            print(name)
            y1,x2,y2,x1 = faceLoc
            y1,x2,y2,x1 = y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
    
    cv2.imshow("webcam", img)
    cv2.waitKey(1)

# imgElon = face_recognition.load_image_file("images/elon_musk.jpg")
# imgElon = cv2.cvtColor(imgElon, cv2.COLOR_BGR2RGB)
# imgTest = face_recognition.load_image_file("images/bill.jpg")
# imgTest = cv2.cvtColor(imgTest, cv2.COLOR_BGR2RGB)
