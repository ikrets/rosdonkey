#!/usr/bin/env python

# This node is a brigde between ROS and a Python 3 
# script that is inferencing on the Edge TPU.

# Edge TPU Python API is Python 3 only and ROS is
# Python 2 only.

import cv2

import rospy
import rospkg
from PIL import Image
import numpy as np
from io import BytesIO
from sensor_msgs.msg import CompressedImage
from donkey_actuator.msg import DonkeyDrive
from potential_field_steering import compute_path_on_fly, compute_polynom, linear_attract_repel_field
import zmq
from controller import SmoothingController

import sys
sys.path.append(rospkg.RosPack().get_path('dataset_utils'))

from transform import make_undistort_birdeye

if __name__ == "__main__":
    undistort_birdeyeview = make_undistort_birdeye(
        input_shape=(320, 240),
        target_shape=(64, 96))

    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    port = rospy.get_param('port', 19090)
    socket.connect('tcp://127.0.0.1:{}'.format(port))

    rospy.init_node("lane_inference_bridge")
    error_decay = rospy.get_param('error_decay', 1)
    gain = rospy.get_param('gain', 1)

    decoded_img = np.empty((240, 320, 3), dtype=np.uint8)
    transformed_img = np.empty((64, 96, 3), dtype=np.uint8)
    resized_lanes = np.empty((32, 48), dtype=np.uint8)

    field = lambda x: linear_attract_repel_field(x,
                                                 attract_level=10,
                                                 attract_value=13,
                                                 attract_width=6,
                                                 repel_level=-100,
                                                 repel_width=3)

    controller = SmoothingController(K=np.exp([-error_decay * i for i in range(5)]),
                                     gain=gain)

    current_ticks_without_throttle = 0
    max_ticks_without_throttle = 2

    def callback(img):
        global current_ticks_without_throttle

        decoded_img = cv2.imdecode(np.fromstring(img.data, np.uint8), 1)
        # The camera has a weird white line on the right edge.
        decoded_img[:, -2:, :] = 0
        undistort_birdeyeview(decoded_img, dst=transformed_img, undistort_interpolation=cv2.INTER_LINEAR)

        socket.send(transformed_img.tobytes())
        img_bytes = socket.recv()

        img = np.frombuffer(img_bytes, dtype=np.bool).reshape((64, 96))
        cv2.resize(img.astype(np.uint8), (48, 32), dst=resized_lanes)

        path, exit_point = compute_path_on_fly(resized_lanes, field, ignore_border=5, steps=2)

        drive = DonkeyDrive()
        drive.source = 'simple_steering'
        max_steering_angle = np.pi / 6

        if exit_point[1] < 12:
            exit = 'left'
        elif exit_point[1] > 36:
            exit = 'right'
        else:
            exit = 'center'

        if current_ticks_without_throttle == max_ticks_without_throttle and exit != 'center':
            exit = 'center'
            current_ticks_without_throttle = 0

        if exit == 'center':
            polynom = compute_polynom(path, deg=2)
            drive.steering = controller.control(target=np.clip((10 * polynom[0] + polynom[1]) / max_steering_angle, -1, 1),
                                                current=0)
            drive.use_constant_throttle = True
        else:
            current_ticks_without_throttle += 1

        if exit == 'stuck':
            polynom = compute_polynom(path, deg=2)
            drive.steering = controller.control(target=np.clip((10 * polynom[0] + polynom[1]) / max_steering_angle, -1, 1),
                                                current=0)
            drive.use_constant_throttle = False
            drive.throttle = 0
        if exit == 'left':
            drive.steering = controller.control(target=-1, current=0)
            drive.use_constant_throttle = False
            drive.throttle = 0
        if exit == 'right':
            drive.steering = controller.control(target=1, current=0)
            drive.use_constant_throttle = False
            drive.throttle = 0

        drive_publisher.publish(drive)

        vis = path[:, :, np.newaxis] * np.array([255, 0, 0], dtype=np.uint8)[np.newaxis, np.newaxis, :]
        vis += resized_lanes[:, :, np.newaxis] * np.array([0, 255, 0], dtype=np.uint8)[np.newaxis, np.newaxis, :]

        with BytesIO() as fp:
            Image.fromarray(vis).save(fp, format='PNG')

            img_msg = CompressedImage()
            img_msg.format = 'png'
            img_msg.data = fp.getvalue()
            publisher.publish(img_msg)

    subscriber = rospy.Subscriber("/raspicam_node/image/compressed", 
	CompressedImage, callback, queue_size=1)
    publisher = rospy.Publisher('/lane_segmentation/image/compressed', CompressedImage,
            queue_size=1)
    drive_publisher = rospy.Publisher('/donkey_drive', DonkeyDrive,
            queue_size=1)

    rospy.spin()
