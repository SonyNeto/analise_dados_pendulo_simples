import numpy as np
import cv2

cap = cv2.VideoCapture("gif.gif")

while True:
    ret, frame = cap.read()

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) == ord('q'):
        break
cv2.imwrite('imagem.png',frame)
cap.release()
cv2.destroyAllWindows()
