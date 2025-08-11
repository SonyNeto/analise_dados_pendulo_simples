import cv2 as cv
import sys
import numpy as np
import tkinter as tk
from PIL import ImageTk, Image
# initialize the camera
cam = cv.VideoCapture("aaa.avi")   # 0 -> index of camera
s, img = cam.read()
if s:    # frame captured without any errors
    #namedWindow("cam-test", WINDOW_OPENGL)
    #namedWindow("cam-test",CV_WINDOW_AUTOSIZE)
    cv.namedWindow("cam-test", cv.WINDOW_AUTOSIZE)
    cv.imshow("cam-test",img)
    cv.waitKey(0)
    cv.destroyWindow("cam-test")
    cv.imwrite("pendulo.png",img) #save image

# -*- coding: utf-8 -*-
# importe imagem
imgCV = cv.imread("pendulo.png")
print(imgCV.shape)
root = tk.Tk()
geometry = "%dx%d+0+0"%(imgCV.shape[0], imgCV.shape[1])
root.geometry()
# convert color from BGR to HSV color scheme
hsv = cv.cvtColor(imgCV, cv.COLOR_BGR2HSV)

def leftclick(event):
    #print("left")
    #print root.winfo_pointerxy()
    print (event.x, event.y)
    #print("BGR color")
    #print (imgCV[event.y, event.x])
    print("Cor HSV:", hsv[event.y, event.x])
# import image
img = ImageTk.PhotoImage(Image.open("pendulo.png"))
panel = tk.Label(root, image = img)
panel.bind("<Button-1>", leftclick)
#panel.pack(side = "bottom", fill = "both", expand = "no")
panel.pack(fill = "both", expand = 1)
root.mainloop() 
