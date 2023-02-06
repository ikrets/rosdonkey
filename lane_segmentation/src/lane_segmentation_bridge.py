#!/usr/bin/env python

# This node is a brigde between ROS and a Python 3 
# script that is inferencing on the Edge TPU.

# Edge TPU Python API is Python 3 only and ROS is
# Python 2 only.

import rospy
from PIL import Image
import numpy as np
from io import BytesIO
from sensor_msgs.msg import CompressedImage
from pwm_joy_teleop.msg import DonkeyDrive
from potential_field_steering import compute_path_on_fly, compute_polynom
import zmq

if __name__ == "__main__":
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect('tcp://127.0.0.1:19090')

    rospy.init_node("lane_inference_bridge")
    rospy.get_param('port', 19090)

    def callback(img):
        # Converting image -> array here as there are weird problems
        # with Pillow in Python 3
        with BytesIO(img.data) as fp:
            img = Image.open(fp).convert('RGB')
            img = np.array(img)

	socket.send(img.tobytes())

	img_bytes = socket.recv()
        img = np.frombuffer(img_bytes, dtype=np.bool)
        img = img.reshape((32, 48))

        path = compute_path_on_fly(img)
        polynom = compute_polynom(path)

        vis = path[:, :, np.newaxis] * np.array([0, 0, 255], dtype=np.uint8)[np.newaxis, np.newaxis, :]
        vis += img[:, :, np.newaxis] * np.array([0, 255, 0], dtype=np.uint8)[np.newaxis, np.newaxis, :]

        poly_x = np.arange(32)
        poly_y = np.polyval(polynom, poly_x).astype(np.uint8)
        inside = (poly_y >= 0) & (poly_y < 48)

        vis[poly_x[inside], poly_y[inside], :] = np.array([255, 0, 0], dtype=np.uint8)

        with BytesIO() as fp:
            Image.fromarray(vis).save(fp, format='PNG')

            img_msg = CompressedImage()
            img_msg.format = 'png'
            img_msg.data = fp.getvalue()
            publisher.publish(img_msg)

    subscriber = rospy.Subscriber("/raspicam_node/image/compressed", 
	CompressedImage, callback, queue_size=1)
    publisher = rospy.Publisher('/lane_segmentation/image/compressed', CompressedImage)

    rospy.spin()