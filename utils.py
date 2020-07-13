import cv2
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh

POSE_MODEL = 'mobilenet_v2_large'  # 'mobilenet_thin'

Nose = 0
Neck = 1
RShoulder = 2
RElbow = 3
RWrist = 4
LShoulder = 5
LElbow = 6
LWrist = 7
RHip = 8
RKnee = 9
RAnkle = 10
LHip = 11
LKnee = 12
LAnkle = 13
REye = 14
LEye = 15
REar = 16
LEar = 17
Background = 18


def point_to_line(p1=np.array([5, 8]), line=(np.array([0, 0]), np.array([10, 10]))):
    p2, p3 = line
    return abs(np.cross(p2 - p1, p3 - p1) / np.linalg.norm(p2 - p1))


def draw_points(img, humans):
    height, width = img.shape[:2]

    for human_idx, human in enumerate(humans):
        h_dic = human.body_parts
        for key, value in h_dic.items():
            x = int(width * value.x)
            y = int(height * value.y)
            img[y:y + 5, x:x + 5] = np.array([0, 255, 0])
            cv2.putText(img, str(human_idx) + '-' + str(key) + '-' + str(round(value.score, 2)), (x, y - 2 * 2),
                        cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 2)

    # x = int(width * loc[0])
    # y = int(height * loc[1])
    # img[y:y + 5, x:x + 5] = np.array([0, 255, 0])
    # return img


def dump():
    estimator = TfPoseEstimator(get_graph_path(POSE_MODEL), target_size=(432, 368),
                                trt_bool=False)
    path = '/home/yufei/PycharmProjects/scanner/images/*'
    for file in glob.glob(path):
        bgr_img = cv2.imread(file)
        humans = estimator.inference(bgr_img, resize_to_default=True, upsample_size=4.0)
        draw_points(bgr_img, humans)
        cv2.imwrite(os.path.join('./pose_img_v2', file.split('/')[-1]), bgr_img)


if __name__ == '__main__':
    print(point_to_line(np.array([5, 1])))
    # estimator = TfPoseEstimator(get_graph_path(POSE_MODEL), target_size=(432, 368),
    #                             trt_bool=False)
    # path = '/home/yufei/PycharmProjects/scanner/images/*'
    # for file in glob.glob(path):
    #     bgr_img = cv2.imread(file)
    #     humans = estimator.inference(bgr_img, resize_to_default=True, upsample_size=4.0)
    #     draw_points(bgr_img, humans)
    #     cv2.imwrite(os.path.join('./pose_img_v2', file.split('/')[-1]), bgr_img)

    # bgr_img = cv2.imread('/home/yufei/PycharmProjects/scanner/images/1592966690.6110272.jpg')
    # humans = estimator.inference(bgr_img, resize_to_default=True, upsample_size=4.0)
    #
    # draw_points(bgr_img, humans)

    # Window name in which image is displayed
    # window_name = 'image'
    #
    # # Using cv2.imshow() method
    # # Displaying the image
    # cv2.imshow(window_name, bgr_img)
    #
    # # waits for user to press any key
    # # (this is necessary to avoid Python kernel form crashing)
    # cv2.waitKey(0)
    #
    # # closing all open windows
    # cv2.destroyAllWindows()
