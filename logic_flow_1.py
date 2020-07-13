"""
拍摄照片的正反面问题，可能对肢体的左右获取值有影响？ 运算
"""
import cv2
import os
import time
import dlib
import numpy as np
import subprocess
import matplotlib.pyplot as plt
import argparse
import logging
import traceback
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
detector = dlib.get_frontal_face_detector()
PART_SCORE_THRESHOLD = 0.2
HUAM_SCORE_THRESHOLD = 1.0
COEFFICIENT = 0.5
DELTA = 1e-5  # 避免分母为０

SHOULDER_ELBOW_DEGREE = 30  # 30 degrees
# 设为0.3，一般人体在镜头中 score>=0.7, 有一部分出镜的话，也大于0.2.
# 小于0.2可以认为，压根就不存在人，假阳性。
HUMAN_EXIST_OR_NOT_THRESHOLD = 0.2

# 语音播放字典
MP3_DIC = {'welcome': 1,
           'poseMaking': 2,
           'scanning': 3,
           'end': 4}


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


def test_human_detect():
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


def face_detect(cap, detector):
    """
    :param detector: dlib face detector
    :param cap: camera
    :return: whether human is inside the room
    """
    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            print('ERROR: FAILED TO CAPTURE IMAGE')
            return  # return none == false!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! how to deal with it when it happens?????????
        if frame is not None:

            cv2.imshow('window', frame)  # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!11
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            dets = detector(img, 0)

            if len(dets) == 0:
                continue
            else:
                if len(dets) == 1:
                    print('有一个人进入')

                else:
                    print('有多个人进入')
                for i, d in enumerate(dets):
                    # print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
                    #     i, d.left(), d.top(), d.right(), d.bottom()))
                    cv2.rectangle(frame, (d.left(), d.top()), (d.right(), d.bottom()), (0, 255, 0), 2)

                cv2.imwrite('./face_crop/' + str(time.time()) + '.jpg', frame)

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
    first_cond = abs(np.arctan(height / (width + DELTA)) * 180 / np.pi) <= SHOULDER_ELBOW_DEGREE  # 肘部抬起，且允许前后调整30度
    print('first_angle: ', abs(np.arctan(height / (width + DELTA)) * 180 / np.pi))
    b = [shoulder.x, elbow.x]
    b.sort()
    second_cond = b[0] <= wrist.x <= b[1]  # 腕部是否在肘部和肩部之间

    ###################################################################################

    shoulder, elbow, wrist = body_parts[3], body_parts[4], body_parts[5]
    a = [(shoulder.x, shoulder.y), (elbow.x, elbow.y)]
    a.sort(key=lambda x: x[0])
    height = (a[1][1] - a[0][1]) * CAP_HEIGHT
    width = (a[1][0] - a[0][0]) * CAP_WIDTH
    third_cond = abs(np.arctan(height / (width + DELTA)) * 180 / np.pi) <= SHOULDER_ELBOW_DEGREE  # 肘部抬起，且允许前后调整30度
    print('third_angle: ', abs(np.arctan(height / (width + DELTA)) * 180 / np.pi))
    b = [shoulder.x, elbow.x]
    b.sort()
    fourth_cond = b[0] <= wrist.x <= b[1]

    if first_cond and second_cond and third_cond and fourth_cond:
        return True
    else:
        return False


# RShoulder, RElbow, RWrist, LShoulder, LElbow, LWrist = res
# 俯视检测  投影模块
def pose_correct_or_not_overlook_projection(
        body_parts):  # 手腕在上面和下面没有考虑。!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    shoulder, elbow, wrist = body_parts[0], body_parts[1], body_parts[2]
    a = [(shoulder.x, shoulder.y), (elbow.x, elbow.y)]
    a.sort(key=lambda x: x[0])
    height = (a[1][1] - a[0][1]) * CAP_HEIGHT
    width = (a[1][0] - a[0][0]) * CAP_WIDTH
    first_cond = abs(np.arctan(height / (width + DELTA)) * 180 / np.pi) <= SHOULDER_ELBOW_DEGREE  # 肘部抬起，且允许前后调整30度
    print('first_angle: ', abs(np.arctan(height / (width + DELTA)) * 180 / np.pi))
    b = [shoulder.x, elbow.x]
    b.sort()
    second_cond = b[0] <= wrist.x <= b[1]  # 腕部是否在肘部和肩部之间

    ###################################################################################

    shoulder, elbow, wrist = body_parts[3], body_parts[4], body_parts[5]
    a = [(shoulder.x, shoulder.y), (elbow.x, elbow.y)]
    a.sort(key=lambda x: x[0])
    height = (a[1][1] - a[0][1]) * CAP_HEIGHT
    width = (a[1][0] - a[0][0]) * CAP_WIDTH
    third_cond = abs(np.arctan(height / (width + DELTA)) * 180 / np.pi) <= SHOULDER_ELBOW_DEGREE  # 肘部抬起，且允许前后调整30度
    print('third_angle: ', abs(np.arctan(height / (width + DELTA)) * 180 / np.pi))
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
    # cv2.namedWindow('window', cv2.WINDOW_NORMAL)

    cnt = 0

    while cap.isOpened():
        cnt += 1
        if cnt % 60 == 0:
            speech(MP3_DIC['poseMaking'])  # 6 seconds
            cnt = 0
            time.sleep(6)  # 给客户６秒钟时间调整姿势，再进行检测, 是否语音要缩短？？？？？？？？？？？？？？？？？？？？？？？？？

        print("*" * 30)
        # ret, img = cap.read()

        # clear cache
        for i in range(5):  # cam property: buffer size = 4, if set 5, it will read the current frame
            ret, img = cap.read()

        if not ret:
            print('ERROR: FAILED TO CAPTURE IMAGE')
            return  # return none == false!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! how to deal with it when it happens?????????
        if img is not None:

            # borrowed from run_webcam.py
            humans = estimator.inference(img, resize_to_default=True, upsample_size=4.0)
            print(humans)
            print(*[h.score for h in humans])

            draw_points(img, humans)
            cv2.imshow('window', img)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break

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
                        cv2.imwrite('./keypoints/' + str(time.time()) + '.jpg', img)
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


def human_exist_or_not(cap, estimator):
    while cap.isOpened():

        print('hi')
        for i in range(5):  # cam property: buffer size = 4, if set 5, it will read the current frame
            ret, img = cap.read()
        # ret, img = cap.read()
        if not ret:
            print('ERROR: FAILED TO CAPTURE IMAGE')
            return  # return none == false!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! how to deal with it when it happens?????????
        if img is not None:

            # borrowed from run_webcam.py
            humans = estimator.inference(img, resize_to_default=True, upsample_size=4.0)

            # human.score ? part.score?
            if len(humans) == 0:
                print('没有检测到人体')
                cv2.imwrite('没有检测到人体' + str(time.time()) + '.jpg', img)
                return
            else:
                scores = np.array([h.score for h in humans])
                res = (scores >= HUMAN_EXIST_OR_NOT_THRESHOLD)  # 0.2
                print(scores)
                if res.any():
                    # 持续提醒离开
                    speech(MP3_DIC['end'])  # 4 seconds 暂停模块，避免持续重复发声！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！１
                    draw_points(img, humans)
                    cv2.imwrite(str(time.time()) + '.jpg', img)
                    time.sleep(4)
                    continue
                else:
                    print('检测到人,但是人的分数极小，可能人已经出镜了，视为假阳性')
                    return
            # elif len(humans) == 1:
            #     target = humans[0]


def test_human_exist_or_not():
    cap = cv2.VideoCapture(0)
    cap.set(3, CAP_WIDTH)
    cap.set(4, CAP_HEIGHT)
    # cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    human_exist_or_not(cap, estimator)


def speech(code):
    """
    开辟新的同步进程，一边发音，一边继续检测。
    注意姿态规范，提示离开
    :return:
    """
    completed = subprocess.Popen(['python', 'voice_out.py', str(code)])


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

                if face_detect(cap, detector):
                    # if human_detect(cap):
                    # 启动语音提示, 欢迎光临，按图示摆好姿势。
                    speech(MP3_DIC['welcome'])  # 7 seconds

                    # 缓一缓???
                    # time.sleep(3s)
                    if human_pose(cap, estimator):  # 如果一直循环不出来，　一段时间后提示，请摆好姿势
                        # 若姿态正确，语音提示受检人固定姿态X秒进行扫描。
                        speech(MP3_DIC['scanning'])

                        time.sleep(5)  # 进行5s中扫描

                        # 检测扫描一段时间,提醒离开
                        speech(MP3_DIC['end'])  #

                        time.sleep(5)  # 给人反应时间

                        # 提醒离开
                        human_exist_or_not(cap, estimator)

                    else:
                        # recheck, 语音提示
                        print('fatal!!!')
                        continue
                else:
                    print("error!!!")
                    continue

        cap.release()
        cv2.destroyAllWindows()

    except Exception as e:
        # 这个是输出错误类别的，如果捕捉的是通用错误，其实这个看不y出来什么
        print('str(Exception):\t', str(Exception))  # 输出  str(Exception):	<type 'exceptions.Exception'>
        # 这个是输出错误的具体原因，这步可以不用加str，输出
        print('str(e):\t\t', str(e))  # 输出 str(e):		integer division or modulo by zero
        print('repr(e):\t', repr(e))  # 输出 repr(e):	ZeroDivisionError('integer division or modulo by zero',)
        print('traceback.print_exc():')
        # 以下两步都是输出错误的具体位置的
        traceback.print_exc()
        print('traceback.format_exc():\n%s' % traceback.format_exc())

    #
    # except:
    #     print('cap is wrong')


if __name__ == '__main__':
    # test_human_detect()
    # test_human_exist_or_not()
    main()

    # speech(2)
