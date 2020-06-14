import os
from PIL import Image
import numpy as np
import cv2
import pickle

BASE_PATH = os.path.dirname(os.path.abspath(__file__))
dir_gambar = os.path.join(BASE_PATH, "misc")

face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
pengenal = cv2.face.LBPHFaceRecognizer_create() 

currId = 0
labelId = {}
y_labels = []
x_train = []

for root, dirs, files in os.walk(dir_gambar):
    for file in files:
        if(file.endswith("png") or file.endswith("jpg")):
            address = os.path.join(root, file)
            label = os.path.basename(os.path.dirname(address)).replace(" ", "-").lower()
            #print(label, address)
            if(not label in labelId):
                labelId[label] = currId
                currId += 1
            id_ = labelId[label]
            #print(labelId)
            img_pil = Image.open(address).convert("L")
            uk = (600,600)
            fin_img = img_pil.resize(uk, Image.ANTIALIAS)
            img_arr = np.array(fin_img, "uint8")
            print(img_arr)
            muka = face_cascade.detectMultiScale(img_arr, scaleFactor = 1.5, minNeighbors = 5)

            for(x,y,w,h) in muka:
                reg_of_interest = img_arr[y:y+h, x:x+w]
                x_train.append(reg_of_interest)
                y_labels.append(id_)

print(y_labels)
print(x_train)

with open("labels.pkl", 'wb') as fi:
    pickle.dump(labelId, fi)

pengenal.train(x_train, np.array(y_labels))
pengenal.save("trainer.yml")

etr = input("Press enter to exit")