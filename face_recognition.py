import cv2
import numpy as np
import os
import boto3
import json
from collections import OrderedDict
from datetime import datetime

data = dict()
group_data = dict()


with open('log.json', 'r+') as file:
    group_data = json.load(file)

""" with open('log.json', 'w+') as file:
    json.dump(group_data, file, ensure_ascii=False, indent='\t') """

s3 = boto3.client('s3')
bucket_name = 'new-technology-project'

#a = s3.get_object(Bucket=bucket_name, Key='log.json')


recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')
cascadePath = "haarcascades/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)
font = cv2.FONT_HERSHEY_SIMPLEX

#iniciate id counter
id = 0

# names related to ids: example ==> loze: id=1,  etc
# 이런식으로 사용자의 이름을 사용자 수만큼 추가해준다.
names = []
control = []
unknown = 0
with open('user.txt', 'r+') as file:
    name = file.readline()
    while name != '':
        names.append(name[:-1])
        control.append(0)
        name = file.readline()

# Initialize and start realtime video capture
cam = cv2.VideoCapture(0)
cam.set(3, 640) # set video widht
cam.set(4, 480) # set video height

# Define min window size to be recognized as a face
minW = 0.1*cam.get(3)
minH = 0.1*cam.get(4)
with open('log.json', 'w+') as make_file:
    while True:
        ret, img =cam.read()
        #img = cv2.flip(img, -1) # Flip vertically
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        
        faces = faceCascade.detectMultiScale( 
            gray,
            scaleFactor = 1.2,
            minNeighbors = 5,
            minSize = (int(minW), int(minH)),
        )

        for(x,y,w,h) in faces:
            cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
            id, confidence = recognizer.predict(gray[y:y+h,x:x+w])
            # Check if confidence is less them 100 ==> "0" is perfect match
            if (confidence < 100):
                #data = json.loads(s)   
                if control[id] == 0:
                    control[id] = 1
                    data['name'] = names[id]
                    data['in_time'] = '%s' % (datetime.now())
                    data['success'] = 'Y'
                    group_data.append(data)
                    json.dump(group_data, make_file, ensure_ascii=False, indent='\t')
                id = names[id]
                confidence = "  {0}%".format(round(100 - confidence))
            else:
                if unknown == 0:
                    unknown = 1
                    data['name'] = 'unknown'
                    data['in_time'] = '%s' % (datetime.now())
                    data['success'] = 'N'
                    group_data.append(data)
                    json.dump(group_data, make_file, ensure_ascii=False, indent='\t')
                id = "unknown"
                confidence = "  {0}%".format(round(100 - confidence))
            
            cv2.putText(img, str(id), (x+5,y-5), font, 1, (255,255,255), 2)
            cv2.putText(img, str(confidence), (x+5,y+h-5), font, 1, (255,255,0), 1)  
        
        cv2.imshow('camera',img) 
        k = cv2.waitKey(10) & 0xff # Press 'ESC' for exiting video
        if k == 27:
            break
# Do a bit of cleanup
print("\n [INFO] Exiting Program and cleanup stuff")
s3.upload_file('log.json', bucket_name, 'log.json', ExtraArgs={'ACL':'public-read'})
cam.release()
cv2.destroyAllWindows()