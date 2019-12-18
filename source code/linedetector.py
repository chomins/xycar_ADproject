# -*- coding: utf-8 -*-

import rospy
import cv2
import math
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


class LineDetector:

    def __init__(self, topic):
        self.text = "Init"

        self.roi_vertical_pos = 295
        self.scan_height = 20
        self.image_width = 640

        self.line_detect_left = 300
        # 왼쪽에서 몇 픽셀 이내에 있는 것만 왼쪽 차선으로 인식할지
        self.line_detect_right = self.image_width - self.line_detect_left
        # 왼쪽에서 몇 픽셀 이상에 있는 것만 오른쪽 차선으로 인식할지

        self.line_before_grad_mean_left = 155
        self.line_before_grad_mean_right = -155
        # 이전 프레임에서의 왼쪽 차선과 오른쪽 차선의 평균 각도
        self.line_grad_admit_range = 20
        # 직선의 각도가 이전 프레임 차선 평균 각도와 얼마 이하 차이나야 값을 인정할지

        self.line_before_xpos_mean_left = 70
        self.line_before_xpos_mean_right = 570
        # 이전 프레임에서의 왼쪽 차선과 오른쪽 차선의 위치
        self.line_xpos_admit_range = 35
        # 직선의 위치가 이전 프레임 차선 위치와 얼마 이하 차이나야 값을 인정할지

        self.line_lost_cnt_left = 11
        self.line_lost_cnt_right = 11
        # 왼쪽 오른쪽 차선이 몇 프레임 연속으로 감지되지 않는지
        self.line_lost_max = 8
        # 몇 프레임 연속으로 감지가 되지 않으면 차선 잃어버림으로 인식할지
        self.line_lost_and_find_admit_range_grad = 100
        # 차선을 잃어버렸을 때, 마지막으로 봤던 각도와 얼마 이하 차이나야 다시 차선이라고 인식할지
        self.line_lost_and_find_admit_range_xpos = 80
        # 차선을 잃어버렸을 때, 마지막으로 봤던 위치와 얼마 이하 차이나야 다시 차선이라고 인식할지

        self.line_return_lposes = [70 for _ in range(2)]
        self.line_return_rposes = [570 for _ in range(2)]
        self.real_lpos = 70
        self.real_rpos = 570
        # 평균을 내서 차선 감지가 조금 부드럽게 되도록

        # Initialize various class-defined attributes, and then...
        self.cam_img = np.zeros(shape=(480, 640, 3), dtype=np.uint8)
        self.mask = np.zeros(shape=(self.scan_height, self.image_width),
                             dtype=np.uint8)
        self.edge = np.zeros(shape=(self.scan_height, self.image_width),
                             dtype=np.uint8)
        self.bridge = CvBridge()
        rospy.Subscriber(topic, Image, self.conv_image)

        self.recoder = cv2.VideoWriter(
            'rec.avi',
            cv2.VideoWriter_fourcc(*'MJPG'),
            30,
            (640, 480)
        )
        print("Start")

    def callback_imu(self, data):
       status = data.status[0].values
       self.roll = status[0].value
       self.pitch = status[1].value
       self.yaw = status[2].value

    def conv_image(self, data):
        self.cam_img = self.bridge.imgmsg_to_cv2(data, 'bgr8')
        v = self.roi_vertical_pos
        self.roi = self.cam_img[v:v + self.scan_height, :]

        hsv = cv2.cvtColor(self.roi, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(self.roi, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        self.edge = cv2.Canny(blur, 60, 70)
        # cv2.rectangle(self.cam_img, (408, 30), (520, 160), (0, 0, 255), 2)
        self.detect_lines()
        self.recoder.write(self.cam_img)

    def detect_lines(self):
        # Return positions of left and right lines detected.
        length = 70
        cv2.line(self.cam_img, (30, 30), (30 + length, 30), (0, 255, 0), 3)
        # 길이가 어느정도인지 가늠하기 위한 코드. length 픽셀이면 화면에서는 어느 정도일까?

        lines = cv2.HoughLinesP(self.edge, 1, np.pi/180, 5, None, 20, 5)
        lines2 = cv2.HoughLinesP(self.edge, 1, np.pi/180, 5, None, 20, 5)

        if lines is not None:
            if lines2 is not None:
                lines = np.hstack([lines, lines2])
        else:
            if lines2 is not None:
                lines = lines2
        # 직선을 찾아주는 신비로운 코드.
        # 차선이 전체적으로는 곡선일 때가 있지만, 세로로 작은 범위를 roi로 지정했기 때문에 직선으로 잘 인식된다.
        # 일정 범위 이내의 각도(기울기)를 지닌 직선을 차선으로 인식하도록 할 예정.

        lines_left_gradient = []  # 왼쪽에 위치한 선들의 기울기(를 각도로 변환한 값)를 저장
        lines_right_gradient = []  # 오른쪽에 위치한 선들의 기울기(를 각도로 변환한 값)를 저장

        lines_left_xpos = []  # 왼쪽에 위치한 선들의 x좌표를 저장 (x1과 x2의 평균)
        lines_right_xpos = []  # 오른쪽에 위치한 선들의 x좌표를 저장 (x1과 x2의 평균)

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                grad = math.degrees(math.atan2((y1 - y2), (x1 - x2)))
                # 직선 기울기를 각도로 변환한 값을 저장.
                xpos = (x1 + x2) / 2
                # x좌표

                is_line = True

                if abs(self.line_before_xpos_mean_left - xpos) < self.line_lost_and_find_admit_range_xpos:  # 왼쪽에 있는 선인가?
                    line_lost_cnt = self.line_lost_cnt_left
                    line_before_grad_mean = self.line_before_grad_mean_left
                    line_before_xpos_mean = self.line_before_xpos_mean_left
                    lines_gradient = lines_left_gradient
                    lines_xpos = lines_left_xpos
                    yes_color = (0, 255, 0)  # 초록색으로 나타냄

                elif abs(self.line_before_xpos_mean_right - xpos) < self.line_lost_and_find_admit_range_xpos:  # 오른쪽에 있는 선인가?
                    line_lost_cnt = self.line_lost_cnt_right
                    line_before_grad_mean = self.line_before_grad_mean_right
                    line_before_xpos_mean = self.line_before_xpos_mean_right
                    lines_gradient = lines_right_gradient
                    lines_xpos = lines_right_xpos
                    yes_color = (255, 0, 0)  # 파란색으로 나타냄

                else:
                    cl = (0, 0, 255)  # 이도저도 아닌 선은 빨간색으로 나타냄
                    is_line = False

                if is_line:
                    if line_lost_cnt >= self.line_lost_max:
                        grad_admit_range = self.line_lost_and_find_admit_range_grad
                        xpos_admit_range = self.line_lost_and_find_admit_range_xpos
                    else:
                        grad_admit_range = self.line_grad_admit_range
                        xpos_admit_range = self.line_xpos_admit_range

                    grad_check = abs(line_before_grad_mean - grad) <= grad_admit_range
                    xpos_check = abs(line_before_xpos_mean - xpos) <= xpos_admit_range
                    if grad_check and xpos_check:
                        lines_gradient.append(grad)
                        lines_xpos.append(xpos)
                        cl = yes_color  # 초록색으로 나타냄
                    else:
                        # if not grad_check:
                            # print("Wrong Grad: ", line_before_grad_mean, grad, line_before_xpos_mean, xpos)
                        # if not xpos_check:
                            # print("Wrong xpos: ", line_before_grad_mean, grad, line_before_xpos_mean, xpos)
                        cl = (0, 0, 255)  # 각도가 너무 차이나므로 빨간색으로 나타냄

                # cv2.line(self.roi, (x1, y1), (x2, y2), cl, 2)  # 선 그리기

        # np.mean은 평균을 구한다.
        # np.std는 표준편차를 구한다.
        # 표준편차가 일정량 이상이면 그냥 값을 무시할 예정 (이전 프레임의 값을 이용하도록)
        if lines_left_gradient:
            left_mean = np.mean(lines_left_gradient)
            self.line_before_grad_mean_left = left_mean

            left_std = np.std(lines_left_gradient)

            left_xpos_int = int(np.mean(lines_left_xpos))
            self.line_before_xpos_mean_left = left_xpos_int

            # cv2.rectangle(self.roi, (left_xpos_int - 35, 0), (left_xpos_int + 35, 20), (255, 255, 255), 2)

            left_mean = "%0.3f" % left_mean
            left_std = "%0.3f" % left_std
            left_xpos = "%0.3d" % left_xpos_int
            self.line_lost_cnt_left = 0
        else:
            left_mean = "None"
            left_std = "None"
            left_xpos = "None"
            self.line_lost_cnt_left += 1

        if lines_right_gradient:
            right_mean = np.mean(lines_right_gradient)
            self.line_before_grad_mean_right = right_mean

            right_std = np.std(lines_right_gradient)

            right_xpos_int = int(np.mean(lines_right_xpos))
            self.line_before_xpos_mean_right = right_xpos_int

            # cv2.rectangle(self.roi, (right_xpos_int - 35, 0), (right_xpos_int + 35, 20), (255, 255, 255), 2)

            right_mean = "%0.3f" % right_mean
            right_std = "%0.3f" % right_std
            right_xpos = "%0.3d" % right_xpos_int
            self.line_lost_cnt_right = 0
        else:
            right_mean = "None"
            right_std = "None"
            right_xpos = "None"
            self.line_lost_cnt_right += 1

        if lines_right_xpos and lines_left_xpos:
            line_dis = "%0.3d" % (right_xpos_int - left_xpos_int)
        else:
            line_dis = "None"

        '''print("| l mean:", left_mean,
              "| l std:", left_std,
              "| l xpos:", left_xpos,
              "| r mean:", right_mean,
              "| r std:", right_std,
              "| r xpos:", right_xpos,
              "| xpos dif:", line_dis,
              )'''

        if lines_left_xpos:
            lpos = left_xpos_int
        else:
            if lines_right_xpos:
                lpos = right_xpos_int - 480
            else:
                lpos = self.line_before_xpos_mean_left

        if lines_right_xpos:
            rpos = right_xpos_int
        else:
            if lines_left_xpos:
                rpos = left_xpos_int + 480
            else:
                rpos = self.line_before_xpos_mean_right

        del self.line_return_lposes[0]
        del self.line_return_rposes[0]
        self.line_return_lposes.append(lpos)
        self.line_return_rposes.append(rpos)

        self.real_lpos = int(np.mean(self.line_return_lposes))
        self.real_rpos = int(np.mean(self.line_return_rposes))

        # cv2.rectangle(self.cam_img,
        #               (self.real_lpos - 35, self.roi_vertical_pos - 10),
        #               (self.real_lpos + 35, self.roi_vertical_pos + self.scan_height + 10),
        #               (0, 255, 0), 2)

        # cv2.rectangle(self.cam_img,
        #               (self.real_rpos - 35, self.roi_vertical_pos - 10),
        #               (self.real_rpos + 35, self.roi_vertical_pos + self.scan_height + 10),
        #               (255, 0, 0), 2)
        # cv2.putText(self.cam_img, self.text, (40, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 200), thickness=2)
        
    def get_line(self):
        return self.real_lpos, self.real_rpos

    def show_images(self, left, right):
        # Display images for debugging purposes;
        # do not forget to call cv2.waitKey().
        pass

    def setText(self, a):
        self.text = a
        self.text += " | " + str(self.yaw)

    def __del__(self):
        self.recoder.release()
        print("Finished")


