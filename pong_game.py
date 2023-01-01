import cvzone
import mediapipe
import numpy as np
import cv2
from cvzone.HandTrackingModule import HandDetector


cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

#importing all images
imgBackground = cv2.imread("Background.png")
imgBall = cv2.imread("Ball.png",cv2.IMREAD_UNCHANGED)
imgGameOver = cv2.imread("gameOver.png")
imgBat1 = cv2.imread("bat1.png",cv2.IMREAD_UNCHANGED)
imgBat2 = cv2.imread("bat2.png",cv2.IMREAD_UNCHANGED)

#Handdetector

detector = HandDetector(detectionCon=0.8,maxHands=2)
#Down I tried to resize the dimension of the images, but it didn't work

'''scale_percent = 70 # percent of original size
width = int(imgBat1.shape[1] * scale_percent / 100)
height = int(imgBat1.shape[1] * scale_percent / 100)
dim = (width, height)

width = int(imgBat2.shape[1] * scale_percent / 100)
height = int(imgBat2.shape[1] * scale_percent / 100)

width = int(imgBall.shape[1] * scale_percent / 100)
height = int(imgBall.shape[1] * scale_percent / 100)

width = int(imgGameOver.shape[1] * scale_percent / 100)
height = int(imgGameOver.shape[1] * scale_percent / 100)

width = int(imgBackground.shape[1] * scale_percent / 100)
height = int(imgBackground.shape[1] * scale_percent / 100)


imgBat1 = cv2.resize(imgBat1, dim, interpolation = cv2.INTER_AREA)
imgBat2 = cv2.resize(imgBat2, dim, interpolation = cv2.INTER_AREA)
imgBall = cv2.resize(imgBall, dim, interpolation = cv2.INTER_AREA)
imgGameOver = cv2.resize(imgGameOver, dim, interpolation = cv2.INTER_AREA)
imgBackground = cv2.resize(imgBackground, dim, interpolation = cv2.INTER_AREA)'''


#variables
ballPosition = [100,100]
speedX= 15
speedY= 15
gameOver = False
score = [0,0]


while True:
    _, img = cap.read()
    imgRaw = img.copy()
    #find hand and its landmark
    img = cv2.flip(img, 1)  # flip the frame horizontally
    hands, img = detector.findHands(img,flipType=False) #flipType=True if you want to flip the image and draw the hand on the frame
    # overlay the background
    img = cv2.addWeighted(img, 0.2, imgBackground, 0.8, 0)

    #check for hand
    if hands:
        for hand in hands:
            x, y, w, h = hand['bbox']
            h1, w1, _ = imgBat1.shape
            #w1 = 26
            #h1 = 129
            y1 = y - h1 // 2
            y1 = np.clip(y1,20,415)
            if hand["type"] == "Left":
                img = cvzone.overlayPNG(img, imgBat1,(59,y1))

                if 59< ballPosition[0] < 59+w1 and y1 < ballPosition[1] < y1+h1:
                    speedX = -speedX
                    ballPosition[0] += 30
                    score[0] += 1

            if hand["type"] == "Right":
                img = cvzone.overlayPNG(img, imgBat2,(1195,y1))
                if 1195-50< ballPosition[0] < 1195 and y1 < ballPosition[1] < y1+h1:
                    speedX = -speedX
                    ballPosition[0] -= 30
                    score[1] += 1

    #game over
    if ballPosition[0] < 40 or ballPosition[0] > 1200:
        gameOver = True

    if gameOver:
        img = imgGameOver
        cv2.putText(img, str(score[1]+score[0]).zfill(2), (585, 360), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (200, 0, 255), 5)

        #if game not over, move the ball
    else:
        #move the ball

        if ballPosition[1] >= 500 or ballPosition[1] <= 10:
            speedY = -speedY

        ballPosition[0] += speedX
        ballPosition[1] += speedY



        #draw the ball
        img = cvzone.overlayPNG(img, imgBall, ballPosition)


        cv2.putText(img, str(score[0]), (300, 650), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 5)
        cv2.putText(img, str(score[1]), (900, 650), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 5)


    img[580:700,20:233] = cv2.resize(imgRaw,(213,120))

    cv2.imshow("img", img)
    key = cv2.waitKey(1)
    if key == ord('r'):
        ballPosition = [100, 100]
        speedX = 15
        speedY = 15
        gameOver = False
        score = [0, 0]
        imgGameOver = cv2.imread("gameOver.png")


