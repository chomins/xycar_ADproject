# -*- coding: utf-8 -*-

import rospy
import cv2
import numpy as np
import pytesseract
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import re

class TrafficSignDetector:

    def __init__(self, topic):
        rospy.init_node('signalDetect')
        rospy.Subscriber(topic, Image, self.conv_image)
        self.sigPub = rospy.Publisher('signal', String, queuesize=1)

        self.finds = []

        stop = ("stop", [
            ['정', '징', '진', '점', '잠', '짐', '청', '칭', '친', '첨', '참', '침', '성', '섬', '심'],
            ['지', '자', '저', '치', '차', '처', '시']
        ])
        self.finds.append(stop)

        slow = ("slow", [
            ['천', '칭', '전', '친', '진', '징'],
            ['천', '칭', '전', '친', '진', '징'],
            ['히', '이', '어', '허', '미', '머']
        ])
        self.finds.append(slow)

        child = ("child", [
            ['어', '허', '이', '히'],
            ['린', '턴', '반', '런'],
            ['이', '미', '허'],
            ['보', '브', '부', '모', '므', '무'],
            ['호', '흐', '후'],
        ])
        self.finds.append(child)

        self.roi_vertical_pos = 30
        self.roi_horizontal_pos = 400
        self.roi_height = 130
        self.roi_width = 120
        self.image_width = 640

        # Initialize various class-defined attributes, and then...
        self.cam_img = np.zeros(shape=(480, 640, 3), dtype=np.uint8)
        self.mask = np.zeros(shape=(self.roi_height, self.roi_width),
                             dtype=np.uint8)
        self.edge = np.zeros(shape=(self.roi_height, self.roi_width),
                             dtype=np.uint8)
        self.bridge = CvBridge()

        self.recoder = cv2.VideoWriter(
            r'/home/dulsik2/Xycar/rec.avi',
            cv2.VideoWriter_fourcc(*'MJPG'),
            30,
            (640, 480)
        )

    def conv_image(self, data):
        print("invoke!")
        self.cam_img = self.bridge.imgmsg_to_cv2(data, 'bgr8')

        v = self.roi_vertical_pos
        w = self.roi_horizontal_pos
        self.roi = self.cam_img[v:v + self.roi_height, w:w + self.roi_width]
        cv2.rectangle(self.cam_img, (self.roi_horizontal_pos, self.roi_vertical_pos),
                      (self.roi_horizontal_pos + self.roi_width, self.roi_vertical_pos + self.roi_height),
                      (0, 255, 255),
                      2)

        self.gray = cv2.cvtColor(self.roi, cv2.COLOR_BGR2GRAY)
        self.detect_signal()
        self.recoder.write(self.cam_img)
        cv2.imshow("asd", self.cam_img)
        cv2.waitKey(0)

    def detect_signal(self):
        circles = cv2.HoughCircles(self.gray, cv2.HOUGH_GRADIENT, 1, 20,
                                   param1=50, param2=40, minRadius=30, maxRadius=45)

        circle_string_finded = False

        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                offset_w = self.roi_horizontal_pos
                offset_h = self.roi_vertical_pos
                cnt = 23
                r = self.roi[i[1] - cnt:i[1] + cnt, i[0] - cnt:i[0] + cnt]
                cv2.circle(self.cam_img, (i[0] + offset_w, i[1] + offset_h), i[2], (255, 0, 0), 1)
                cv2.rectangle(self.cam_img, (i[0] - cnt + offset_w, i[1] - cnt + offset_h),
                              (i[0] + cnt + offset_w, i[1] + offset_h), (0, 255, 0), 1)
                try:
                    self.pub_info = pytesseract.image_to_string(r)
                    if "30" in self.pub_info:
                        self.pub_info = "30"
                        circle_string_finded = True

                    elif "50" in self.pub_info:
                        self.pub_info = "50"
                        circle_string_finded = True
                    else:
                        self.pub_info = "not"

                except Exception as e:
                    print(e)

        if not circle_string_finded:
            self.pub_info = self.tesseract_start("python", self.roi)

    def adThresholding(self, im, val=31, val2=0):
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        mask = cv2.adaptiveThreshold(im, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, val, val2)
        return mask

    def tesseracting(self, name, mask, bitwising):
        if mask is None:
            return ''

        cv2.imshow(name, mask)

        st = pytesseract.image_to_string(mask, lang='kor')

        if bitwising:
            mask_bitwise = np.full(shape=(mask.shape[0], mask.shape[1]), fill_value=255,
                                   dtype=np.uint8)
            mask_bited = np.bitwise_xor(mask, mask_bitwise)
            st += "|" + pytesseract.image_to_string(mask_bited, lang='kor')

        st = st.replace(' ', '')
        st = st.replace('\n', '')
        return st

    def tesseract_start(self, filepath, im):
        ans = []
        ans.append(self.tesseracting(filepath + "1",
                                     self.adThresholding(im, 101, 10),
                                     False))
        return self.detecting(ans)

    def detecting(self, ans):
        word = ans[0]
        word = re.sub('[-=.#/?:$}{]`!%^&*()_+', '', word)
        # print(word)

        for fn in self.finds:
            if self.detectWord(fn[1], word):
                print("==========================================ITS", fn[0])
                return fn[0]

        return "not"

    def detectWord(self, org, word):
        if len(word) < len(org):
            return False

        score = 0

        for i in range(len(org)):
            if word[i] in org[i]:
                score += 1
        if score >= len(org) // 2 + 1:
            return True

        if len(word) > len(org):
            p = len(word) - len(org)
            for i in range(len(org)):
                if word[i + p] in org[i]:
                    score += 1
            if score >= len(org) // 2 + 1:
                return True
            else:
                return False
        else:
            return False

    def pub(self):
        print("pub", self.pub_info)
        self.sigPub.publish(self.pub_info)

    def __del__(self):
        self.recoder.release()
        cv2.destroyAllWindows()

    # def display_msg(self, signal):
    #     msgs = [("50", "최저 속도 제한 표지판 인식됨", 35),
    #             ("30", "최고 속도 제한 표지판 인식됨", 15),
    #             ("slow", "서행 표지판 인식됨", 15),
    #             ("child", "어린이 보호 구역 표지판 인식됨", 15),
    #             ("not", "기본 주행 모드", 25)
    #             ]
    #
    #     for s, msg, speed in msgs:
    #         if s in signal:
    #             cv2.putText(self.cam_img, msgs, (40, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 200), thickness=2)
    #             # cv2.putText(self.cam_img, self.text, (40, 430), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 200), thickness=2)

if __name__ == "__main__":
    rate = rospy.Rate(10)
    detector = TrafficSignDetector('/usb_cam/image_raw')
    while not rospy.is_shutdown():
        detector.pub()
        rate.sleep()

