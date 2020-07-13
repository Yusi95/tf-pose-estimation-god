import cv2
import time
import numpy as np


def inner_loop(cap):
    while cap.isOpened():
        ret, img = cap.read()

        cv2.imshow('inner_window', img)

        res = cv2.waitKey(1) & 0xFF
        if res == ord('i'):
            break

        print("inner window!!!")


def capture_img():
    try:
        cap = cv2.VideoCapture(0)
        # cap.set(3, 1920)  # 设置分辨率  2K
        # cap.set(4, 1080)
        cap.set(3, 1280)
        cap.set(4, 720)

        cv2.namedWindow('window')
        cv2.namedWindow('inner_window')
        while cap.isOpened():
            ret, img = cap.read()

            if not ret:
                print('ERROR: FAILED TO CAPTURE IMAGE')
                return
            if img is not None:
                cv2.imshow('window', img)
                print('outer window')

                res = cv2.waitKey(1000) & 0xFF
                print(res)
                if res == ord('q'):
                    break

                inner_loop(cap)

        # When everything done, release the capture
        cap.release()
        cv2.destroyAllWindows()

    except:
        print('cap is wrong')


if __name__ == '__main__':
    capture_img()
