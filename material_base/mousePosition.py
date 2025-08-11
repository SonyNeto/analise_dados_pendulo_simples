# -*- coding: utf-8 -*-
import cv2 as cv
import numpy as np
import tkinter as tk
from PIL import ImageTk, Image
import sys
# importe imagem
imgCV = cv.imread("imagem.jpg")
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
    print("HSV color")
    print("codigo hsv", hsv[event.y, event.x])
# import image
img = ImageTk.PhotoImage(Image.open("imagem.jpg"))
panel = tk.Label(root, image = img)
panel.bind("<Button-1>", leftclick)
#panel.pack(side = "bottom", fill = "both", expand = "no")
panel.pack(fill = "both", expand = 1)
root.mainloop() 
