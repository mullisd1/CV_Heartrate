import cv2

# Load the cascade
face_cascade = cv2.CascadeClassifier('opencv_xml/haarcascade_frontalface_default.xml')

# To capture video from webcam. 
cap = cv2.VideoCapture(0)
# To use a video file as input 
# cap = cv2.VideoCapture('filename.mp4')

#it's best to collet color samples from cheeks... but if everything is still, does it matter?
#yes cause it keeps slightly changing size
data = []
while True:
    # Read the frame
    _, img = cap.read()
    # Convert to grayscale
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detect the faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    # Draw the rectangle around each face
    for (x, y, w, h) in faces:
        x = int(x*1.2)
        w = int(w*0.6)
        y = int(y*1.2)
        h = int(h*0.6)
        

        total = 0
        count = 0
        for i in range(y, y + h):
            for j in range(x, x + w):
                count = count + 1
                total = total+gray[i][j]
        if(count == 0):
            data.append(0)
        else:
            data.append(total/count)
        
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
    # Display
    cv2.imshow('img', img)
    # Stop if escape key is pressed
    k = cv2.waitKey(30) & 0xff
    if k==27:
        print(data)
        break

    
# Release the VideoCapture object
cap.release()


