#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy, time
import math

from linedetector import LineDetector
from motordriver import MotorDriver
from std_msgs.msg import String


class AutoDrive:

    def __init__(self):
        rospy.init_node('autodrive')
        rospy.Subscriber('/signal', String, self.detect_signal)
        self.line_detector = LineDetector('/usb_cam/image_raw')
        self.driver = MotorDriver('/xycar_motor_msg')
        self.end_time = -1
        self.signal_time = 3
        self.under_signal = "not"

    def trace(self):
        line_l, line_r = self.line_detector.get_line()
        angle = self.steer(line_l, line_r)
        speed = self.accelerate()
        self.driver.drive(angle + 90, speed + 90)

    def steer(self, left, right):
        mid = (left + right) // 2

        angle = (mid - 320) * 0.5
        angle = angle / 50
        isminus = angle < 0

        angle = math.pow(abs(angle), 1.7)
        if isminus:
            angle = -angle

        angle = angle * 50
        if angle > 50:
            angle = 50
        elif angle < -50:
            angle = -50

        return angle

    def accelerate(self):
        if self.under_signal == "stop":
            return 0

        if self.under_signal != "not":
            if self.under_signal == "50":
                return 35
            else:
                return 15
        else:
            return 25

    def detect_signal(self, data):
        if time.time() < self.end_time:
            return

        signal = data

        if signal != "not":
            self.end_time = time.time() + self.signal_time
            self.under_signal = signal
        else:
            self.under_signal = signal

    def exit(self):
        print('finished')


if __name__ == '__main__':
    car = AutoDrive()
    time.sleep(3)
    rate = rospy.Rate(60)
    while not rospy.is_shutdown():
        car.trace()
        rate.sleep()
    rospy.on_shutdown(car.exit)
