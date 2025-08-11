# Escrito por Adriano A. Batista, 2018, com pedaços de código obtidos 
# na internet 
# USAGE
# python ball_tracking.py --video ball_tracking_example.mp4
# python ball_tracking.py

# import the necessary packages
from collections import deque
import datetime as dat
import numpy as np
import argparse
import imutils
import cv2
import time

# Data e hora local
data = str(dat.date.today())
localtime = time.localtime(time.time())
hora = localtime.tm_hour
minuto = localtime.tm_min

if hora<10:
    hora = '0'+str(hora)
else:
    hora = str(hora)
if minuto<10:
    minuto = '0'+str(minuto)
else:
    minuto = str(minuto)

dataHoraMin = data + '_' + hora + 'h' + minuto + 'm_'

# Define nome do arquivo
nomeArquivo = "../Dados/" + dataHoraMin + 'SerieTempPend.csv'
arquivo = open(nomeArquivo, "w")
print(nomeArquivo)

def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
       return v
    return v / norm
# Coordenadas do suporte do pêndulo

x_s = 252
y_s = 124
# Coordenadas da extremidade do pêndulo em repouso
x_ext = 319
y_ext = 242


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the (optional) video file")
ap.add_argument("-b", "--buffer", type=int, default=64, help="max buffer size")
args = vars(ap.parse_args())

# define the lower and upper boundaries of the "green"
# ball in the HSV color space, then initialize the
# list of tracked points
greenLower = (130, 190, 235)
greenUpper = (150, 210, 255)
#pts = deque(maxlen=args["buffer"])

# if a video path was not supplied, grab the reference
# to the webcam
if not args.get("video", False):
    camera = cv2.VideoCapture(1)
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    #fourcc = cv2.cv.CV_FOURCC(*'XVID')
    nomeArquivo = dataHoraMin+'SerieTempPend.avi'
    out = cv2.VideoWriter(nomeArquivo,fourcc, 20.0, (640,480))

# otherwise, grab a reference to the video file
else:
    camera = cv2.VideoCapture(args["video"])


# keep looping
count = 0
while True:
    # grab the current frame
    (grabbed, frame) = camera.read()

    # if we are viewing a video and we did not grab a frame,
    # then we have reached the end of the video
    if args.get("video") and not grabbed:
        break

    # resize the frame, blur it, and convert it to the HSV
    # color space
    frame = imutils.resize(frame, width=640)
    # blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # construct a mask for the color "green", then perform
    # a series of dilations and erosions to remove any small
    # blobs left in the mask
    mask = cv2.inRange(hsv, greenLower, greenUpper)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)


    # find contours in the mask and initialize the current
    # (x, y) center of the ball
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    center = None

    # only proceed if at least one contour was found
    if len(cnts) > 0:
        # find the largest contour in the mask, then use
        # it to compute the minimum enclosing circle and
        # centroid
        c = max(cnts, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        #print('center coordinates', center)
        # only proceed if the radius meets a minimum size
        if radius > 10:
            # draw the circle and centroid on the frame,
            # then update the list of tracked points
            #cv2.circle(frame, (int(x), int(y)), int(radius),(0, 255, 255), 2)
            #cv2.circle(frame, center, 5, (0, 0, 255), -1)
            if (count == 10):
                 t_0 = time.time()
                 print(t_0)
            if (count >= 10):
#                print (time.time()-t_0, x, y)
                #########################################################3
                p1 = np.array([x_s, y_s], dtype = np.float)
                p2 = np.array([x_ext, y_ext], dtype = np.float)
                # coordenada da ponta do pêndulo
                pt = np.array([x, y], dtype = np.float)
                v0_vert = p2 - p1
                n_vert = normalize(v0_vert)
                n_hor = np.array([1, 0])
                lt_vec = pt - p1
                lt_normal = normalize(lt_vec)
                # phi_t: angulo com vertical
                cos_phi_t = np.inner(lt_normal,n_vert)
                sin_phi_t = np.inner(lt_normal,n_hor)
                phi_t = np.arctan2(sin_phi_t, cos_phi_t)

                # Salvar o ângulo
                arquivo.write("%g, %g\n"%(time.time()-t_0, np.degrees(phi_t)))
                #print (time.time()-t_0, np.degrees(phi_t))

                # draw the circle and centroid on the frame,
                # then update the list of tracked points
                cv2.circle(frame, (int(x), int(y)), int(radius),(0, 255, 255), 2)
                #print int(x), int(y)
                cv2.circle(frame, center, 5, (0, 0, 255), -1)
                # cv2.circle(frame, (x_s, y_s), 5, (0, 0, 255), -1)
                #########################################################3
    count += 1


    # update the points queue
#    pts.appendleft(center)

    # loop over the set of tracked points
#    for i in range(1, len(pts)):
        # if either of the tracked points are None, ignore
        # them
#        if pts[i - 1] is None or pts[i] is None:
#            continue

        # otherwise, compute the thickness of the line and
        # draw the connecting lines
#        thickness = int(np.sqrt(args["buffer"] / float(i + 1)) * 2.5)
#        cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness)

    # save and show the frame to our screen
    if not args.get("video", False):
        out.write(frame)
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the 'q' key is pressed, stop the loop
    if key == ord("q"):
        break

# cleanup the camera and close any open windows
arquivo.close()
camera.release()
if not args.get("video", False):
    out.release()
cv2.destroyAllWindows()
