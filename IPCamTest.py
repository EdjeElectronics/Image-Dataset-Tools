# Basic IP camera test
import cv2
import numpy as np
import time

imgW = 1280
imgH = 720

# Initialize IP camera
cap1 = cv2.VideoCapture('rtsp://username:password@192.168.1.25:554')
time.sleep(1)

frame_rate_calc = 1
freq = cv2.getTickFrequency()

while True:

    t1 = cv2.getTickCount()

    hasFrame1, frame1 = cap1.read()
    #if not hasFrame1:
        #print('Info: Missed frame from camera')
        #continue

    frame1 = cv2.resize(frame1, (imgW,imgH))
    cv2.imshow('Test 1',frame1)

    t2 = cv2.getTickCount()
    time1 = (t2-t1)/freq
    frame_rate_calc = 1/time1

    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    elif key == ord('p'):
        #Take a picture!
        name = input('Enter picture name: ')
        cv2.imwrite(name, frame1)

cv2.destroyAllWindows()
cap1.release()
