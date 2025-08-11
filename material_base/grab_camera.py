import cv2 as cv
import sys
# initialize the camera
cam = cv.VideoCapture("gif.gif")   # 0 -> index of camera
while True:
    s, img = cam.read()
   # frame captured without any errors
    #namedWindow("cam-test", WINDOW_OPENGL)
    #namedWindow("cam-test",CV_WINDOW_AUTOSIZE)
    cv.namedWindow("cam-test", cv.WINDOW_AUTOSIZE)
    cv.imshow("cam-test",img)
    cv.waitKey(0)
    cv.destroyWindow("cam-test")
    cv.imwrite('imagem.jpg',img) #save image
