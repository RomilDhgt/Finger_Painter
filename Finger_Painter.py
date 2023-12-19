import cv2 
import time
import os
import numpy as np
import Hand_Tracking_Module as htm

# initializing variables for FPS tracking
pastTime = 0
currentTime = 0

folderPath = "C:\\Users\\Romil\\GIT Projects\\New folder\\Hand_tracking\\Banner"
pathList = os.listdir(folderPath)
overlayList = []

for path in pathList:
    image = cv2.imread(f'{folderPath}/{path}')
    overlayList.append(image)

header = overlayList[0]

drawColor = (255,0,0)
brushThickness = 15
eraserThickness = 50
xp, yp = 0, 0

# Creating the video capture object using my laptops video camera
cap = cv2.VideoCapture(0)

# Setting the width and height of the camera display
cap.set(3,1280)
cap.set(4,720)

detector = htm.handDetector(dCon=0.7, tCon=0.7)

imgCanvas = np.zeros((720,1280,3),np.uint8)

while True:
    # Getting the video from the camera
    success, img = cap.read()

    # flip the image
    img = cv2.flip(img,1)

    # Using findHands function in the Hand_Tracking_Module to overlay hand mesh on hand in the image captured by the camera
    img = detector.findHands(img)
    lmList = detector.findPos(img,draw=False)
    
    if len(lmList) != 0:
        # Getting position of the index and middle fingers
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]
    
    # using the fingers up method to determine which fingers are up
    fingers = detector.fingersUp()
    
    if len(fingers) != 0:
        # Selection mode
        if fingers[1] and fingers[2]:
            xp, yp = 0, 0
            cv2.rectangle(img,(x1,y1-15),(x2,y2+15),drawColor,cv2.FILLED)
            # Selecting the drawing tools
            if y1 < 100:
                if 225<x1<425:
                    header = overlayList[1]
                    drawColor = (0,0,255)
                elif 475<x1<675:
                    header = overlayList[2]
                    drawColor = (0,255,0)
                elif 725<x1<925:
                    header = overlayList[3]
                    drawColor = (255,0,0)
                elif 975<x2<1175:
                    header = overlayList[4]
                    drawColor = (0,0,0)

        # Drawing mode
        if fingers[1] and fingers[2] == False:
            cv2.circle(img, (x1,y1),15,drawColor,cv2.FILLED)
            if xp == 0 and yp == 0:
                xp, yp = x1, y1

            if drawColor == (0,0,0):
                cv2.line(img,(xp,yp),(x1,y1),drawColor,eraserThickness)
                cv2.line(imgCanvas,(xp,yp),(x1,y1),drawColor,eraserThickness)
            else:
                cv2.line(img,(xp,yp),(x1,y1),drawColor,brushThickness)
                cv2.line(imgCanvas,(xp,yp),(x1,y1),drawColor,brushThickness)
           
            xp, yp = x1, y1

    # Creating a gray image from the canvas to get a inverse of the canvas 
    imgGray = cv2.cvtColor(imgCanvas,cv2.COLOR_RGB2GRAY)
    _, imgInv = cv2.threshold(imgGray,25,255,cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv,cv2.COLOR_GRAY2BGR)
    # The and operation of  the inversed image and the image captured will project the inversed image on to the camera image
    img = cv2.bitwise_and(img,imgInv)
    # The or operation will bring in that we are trying to draw with into the camera image 
    img = cv2.bitwise_or(img,imgCanvas)

    # Getting timestamps to be able to use the trackers positioning data
    currentTime = time.time()
    framesPerSec = 1/(currentTime-pastTime)
    pastTime = currentTime
    # Displaying the frames per second the the screen
    cv2.putText(img,f'FPS:{int(framesPerSec)}', (40,680), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,255), 3)

    # Showing the video from camera in a window and setting the header image 
    img[0:100,0:1280] = header
    # img = cv2.addWeighted(img,0.5,imgCanvas,0.5,0)  
    cv2.imshow("Image",img)
    cv2.imshow("Canvas",imgCanvas)
    cv2.waitKey(1)
