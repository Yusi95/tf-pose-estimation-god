"""
拍摄照片的正反面问题，可能对肢体的左右获取值有影响？ 运算
"""
import cv2
import os
import time
import dlib
import numpy as np
import matplotlib.pyplot as plt
import argparse
import logging
import time

from utils import draw_points
from tf_pose import common
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh

# cap.set(3, 1280)
# cap.set(4, 720)
CAP_WIDTH = 1280
CAP_HEIGHT = 720

THRESHOLD = 10
POSE_MODEL = 'mobilenet_thin'
# model loading
estimator = TfPoseEstimator(get_graph_path(POSE_MODEL), target_size=(432, 368),
                            trt_bool=False)
PART_SCORE_THRESHOLD = 0.2
HUAM_SCORE_THRESHOLD = 1.0
COEFFICIENT = 0.5

SHOULDER_ELBOW_DEGREE = 30  # 30 degrees


def human_detect(cap):
    """
    :param cap:
    :return: whether human inside the room
    """
    original_img = cv2.imread('./original.jpg')

    while cap.isOpened():
        ret, img = cap.read()
        cv2.imshow('window', img)
        if not ret:
            print('ERROR: FAILED TO CAPTURE IMAGE')
            return  # return none == false!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! how to deal with it when it happens?????????
        if img is not None:
            diff = np.mean(np.abs(original_img.astype(int) - img.astype(int)))
            print('difference: ', diff)
            if diff >= THRESHOLD:
                return True


def face_detect(cap, detector):
    """
    :param detector: dlib face detector
    :param cap: camera
    :return: whether human is inside the room
    """
    while cap.isOpened():
        ret, frame = cap.read()
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        dets = detector(img, 0)

        if len(dets) == 0:
            continue
        else:
            if len(dets) == 1:
                print('有一个人进入')
            else:
                print('有多个人进入')

            return True


# RShoulder, RElbow, RWrist, LShoulder, LElbow, LWrist = res
# 平视检测
def pose_correct_or_not(body_parts):  # 如果使用需要修改成绝对像素，因为宽和高不相等，不能约掉！！！！！！！！！！！！！！！！！！
    shoulder, elbow, wrist = body_parts[0], body_parts[1], body_parts[2]
    first_cond = ((shoulder.y + elbow.y) / 2) >= (wrist.y + abs(shoulder.x - elbow.x) * COEFFICIENT)
    a = [shoulder.x, elbow.x]
    a.sort()
    second_cond = a[0] <= wrist.x <= a[1]

    shoulder, elbow, wrist = body_parts[3], body_parts[4], body_parts[5]
    thrid_cond = ((shoulder.y + elbow.y) / 2) >= (wrist.y + abs(shoulder.x - elbow.x) * COEFFICIENT)

    a = [shoulder.x, elbow.x]
    a.sort()
    fourth_cond = a[0] <= wrist.x <= a[1]

    if first_cond and second_cond and thrid_cond and fourth_cond:
        return True
    else:
        return False


# RShoulder, RElbow, RWrist, LShoulder, LElbow, LWrist = res
# 俯视检测
def pose_correct_or_not_overlook(body_parts):  # 手腕在上面和下面没有考虑。!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    shoulder, elbow, wrist = body_parts[0], body_parts[1], body_parts[2]
    a = [(shoulder.x, shoulder.y), (elbow.x, elbow.y)]
    a.sort(key=lambda x: x[0])
    height = (a[1][1] - a[0][1]) * CAP_HEIGHT
    width = (a[1][0] - a[0][0]) * CAP_WIDTH
    first_cond = abs(np.arctan(height / width) * 180 / np.pi) <= SHOULDER_ELBOW_DEGREE  # 肘部抬起，且允许前后调整30度
    print('first_angle: ', abs(np.arctan(height / width) * 180 / np.pi))
    b = [shoulder.x, elbow.x]
    b.sort()
    second_cond = b[0] <= wrist.x <= b[1]  # 腕部是否在肘部和肩部之间

    ###################################################################################

    shoulder, elbow, wrist = body_parts[3], body_parts[4], body_parts[5]
    a = [(shoulder.x, shoulder.y), (elbow.x, elbow.y)]
    a.sort(key=lambda x: x[0])
    height = (a[1][1] - a[0][1]) * CAP_HEIGHT
    width = (a[1][0] - a[0][0]) * CAP_WIDTH
    third_cond = abs(np.arctan(height / width) * 180 / np.pi) <= SHOULDER_ELBOW_DEGREE  # 肘部抬起，且允许前后调整30度
    print('third_angle: ', abs(np.arctan(height / width) * 180 / np.pi))
    b = [shoulder.x, elbow.x]
    b.sort()
    fourth_cond = b[0] <= wrist.x <= b[1]

    if first_cond and second_cond and third_cond and fourth_cond:
        return True
    else:
        return False


def human_pose(cap, estimator):
    """
    :param img:
    :return: whether human pose is okay

    RShoulder = 2
    RElbow = 3
    RWrist = 4
    LShoulder = 5
    LElbow = 6
    LWrist = 7
    """
    cv2.namedWindow('window', cv2.WINDOW_NORMAL)

    while cap.isOpened():
        print("*" * 30)
        ret, img = cap.read()
        cv2.imshow('window', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # borrowed from run_webcam.py
        humans = estimator.inference(img, resize_to_default=True, upsample_size=4.0)
        print(humans)

        draw_points(img, humans)
        cv2.imshow('window', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # human.score ? part.score?
        if len(humans) == 0:
            print('没有检测到人体')
            continue
        elif len(humans) == 1:
            target = humans[0]

            dic = target.body_parts
            res = [dic.get(i) for i in range(2, 8)]

            # 虽然检测到人，但是核心部位缺失
            if None in res:
                print('虽然检测到人，但是核心部位缺失')
                continue
            # RShoulder, RElbow, RWrist, LShoulder, LElbow, LWrist = res

            part_score = np.array([part.score for part in res]) >= PART_SCORE_THRESHOLD
            if part_score.all():
                if pose_correct_or_not_overlook(res):
                    print('姿势正确')
                    cv2.imwrite('./ok' + str(time.time()) + '.jpg', img)
                    return True
                else:
                    # 提示姿势矫正
                    print('提示姿势矫正')
                    continue
            else:
                # 有一些关键点置信度过低
                print('有一些关键点置信度过低')
                continue
        else:
            print('检测到多个人体骨骼')
            ## 策略一: continue,作废. 策略二：选取分数高的那个
            continue
        # print(humans)


def test_human_pose():
    cap = cv2.VideoCapture(0)
    cap.set(3, CAP_WIDTH)
    cap.set(4, CAP_HEIGHT)
    human_pose(cap, estimator)


def speech():
    """
    注意姿态规范，提示离开
    :return:
    """


def test():
    try:
        cap = cv2.VideoCapture(0)
        # cap.set(3, 1920)  # 设置分辨率  2K
        # cap.set(4, 1080)
        cap.set(3, 1280)
        cap.set(4, 720)

        cv2.namedWindow('window')

        while cap.isOpened():
            # ret, img = cap.read()

            # if not ret:
            #     print('ERROR: FAILED TO CAPTURE IMAGE')
            #     return
            # if img is not None:
            #     cv2.imshow('window', img)
            if human_detect(cap):
                print('human detect')
                # time.sleep(2)
            # res = cv2.waitKey(1) & 0xFF
            # print(res)
            # if res == ord('q'):
            #     break
            # elif res == ord('p'):
            #     cv2.imwrite('../images/' + str(time.time()) + '.jpg', img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        # When everything done, release the capture
        cap.release()
        cv2.destroyAllWindows()

    except:
        print('cap is wrong')


def main():
    try:
        cap = cv2.VideoCapture(0)
        # cap.set(3, 1920)  # 设置分辨率  2K
        # cap.set(4, 1080)

        # cap.set(3, 1280)
        # cap.set(4, 720)
        cap.set(3, CAP_WIDTH)
        cap.set(4, CAP_HEIGHT)

        cv2.namedWindow('window')

        while cap.isOpened():
            ret, img = cap.read()

            if not ret:
                print('ERROR: FAILED TO CAPTURE IMAGE')
                return
            if img is not None:
                cv2.imshow('window', img)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                if human_detect(cap):
                    if human_pose(cap):
                        # 检测扫描一段时间,提醒离开
                        if human_detect(cap):
                            # 提醒离开
                            pass
                        else:
                            continue
                    else:
                        # recheck, 语音提示
                        pass
                else:
                    continue

        cap.release()
        cv2.destroyAllWindows()

    except:
        print('cap is wrong')


if __name__ == '__main__':
    # test()
    test_human_pose()
