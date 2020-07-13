#!/usr/bin/python3
import cv2
import time

## opening videocapture
cap = cv2.VideoCapture(0)

## some videowriter props
sz = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),  # (640,480)
      int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
# print(sz)
fps = 20
# fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
# fourcc = cv2.VideoWriter_fourcc('m', 'p', 'e', 'g')
fourcc = cv2.VideoWriter_fourcc(*'mpeg')

## open and set props
vout = cv2.VideoWriter()
vout.open('./video/' + str(time.time()) + '.mp4', fourcc, fps, sz, True)
cv2.namedWindow('window')
cnt = 0
while True:
    cnt += 1
    print(cnt)
    _, frame = cap.read()
    cv2.putText(frame, str(cnt), (10, 20), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1, cv2.LINE_AA)
    vout.write(frame)
    cv2.imshow('window', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vout.release()
cap.release()
cv2.destroyAllWindows()
