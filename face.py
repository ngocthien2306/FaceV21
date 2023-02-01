import cv2
import face_recognition as face
import matplotlib.pyplot as plt
import numpy as np


webcam_video_stream = cv2.VideoCapture(0)
faces_location = []
while True: 
    # get current frame form the live video
    ret, current_frame = webcam_video_stream.read()
    # resize to 1/4 to process faster

    # w, h, d = np.array(current_frame).shape
    # print(w, h, d)
    # current_frame =  cv2.resize(current_frame,(120, 160))
    # face_location = face.face_locations(current_frame, model='hog')
    # for index, current_face_location in enumerate(face_location):
    #     print(current_face_location)
    #     (top, right, bottom, left) = current_face_location * 4
    #     print('Found face {} at location T {} R {} B {} L {}'.format(index+1, top, right, bottom, left))
    #     cv2.rectangle(current_frame, (left, top), (right, bottom), (0,0,255), 2)
    print(current_frame)
    cv2.imshow("Webcam", current_frame)
    
    if (cv2.waitKey(1) & 0xFF == ord('q')):
        break
webcam_video_stream.release() 
cv2.destroyAllWindows() 

    
        
    
        