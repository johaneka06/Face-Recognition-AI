import numpy as np
import cv2
import pickle
import datetime as dt

face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
pengenal = cv2.face.LBPHFaceRecognizer_create() 
pengenal.read("trainer.yml")

badge = {"name": 1}
with open("labels.pkl", 'rb') as fi:
    og_badge = pickle.load(fi)
    badge = {v:k for k, v in og_badge.items()}

cap = cv2.VideoCapture(0)

while(True):
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    muka = face_cascade.detectMultiScale(gray, scaleFactor = 1.5, minNeighbors = 5) #haar cascade
    waktu = dt.datetime.now().time()
    for(x, y, w, h) in muka:
        print(x,y,w,h)
        roi_gray = gray[y:y+h, x:x+w] #ROI = Region Of Interest
        roi_color = frame[y:y+h, x:x+w]

        id_, conf = pengenal.predict(roi_gray) #Deep learning
        if (conf >= 35):
            #print(id_)
            print(badge[id_])
            font = cv2.FONT_HERSHEY_SIMPLEX
            nama = badge[id_]
            color = (255,255,255)
            stroke = 2
            namaAndConfussion = nama + " " + str(conf)
            cv2.putText(frame, namaAndConfussion, (x,y), font, 1, color, stroke, cv2.LINE_AA)

        image_item = "my-img.png"
        cv2.imwrite(image_item, roi_gray)

        warna_border = (0,255,0) #Color become green
        lebar_border = 3
        akhir_lebar = x + w
        akhir_tinggi = y + h
        cv2.rectangle(frame, (x,y), (akhir_lebar, akhir_tinggi), warna_border, lebar_border)

    cv2.imshow('frame', frame)
    if(cv2.waitKey(20) & 0xFF == ord('q')):
        break

cap.release()
cv2.destroyAllWindows()