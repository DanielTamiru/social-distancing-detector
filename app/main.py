import cv2

trained_data = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml")


#Capture video from webcam
webcam = cv2.VideoCapture(0)

#Loop indefinetly 
while True:

    #read current frame
    successful_frame_read, frame = webcam.read()
 
    #convert image to gray scale
    grayscaled_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #detect faces
    face_coordinates = trained_data.detectMultiScale(grayscaled_frame)

    #draw rectangles
    for (x, y, w, h) in face_coordinates:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0,255) , 2) 

    #output frame with rectangle drawn
    cv2.imshow('Body Detector', frame)

    #display frame for 1ms
    key = cv2.waitKey(1)
    
    ### QUIT using 'q' key
    if key==81 or key==113:
        break

#release webcam
webcam.release()