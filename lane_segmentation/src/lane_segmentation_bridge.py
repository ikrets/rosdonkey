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

	img_msg = CompressedImage()
	img_msg.format = 'png'
	img_msg.data = img_bytes
	publisher.publish(img_msg)

    subscriber = rospy.Subscriber("/raspicam_node/image/compressed", 
	CompressedImage, callback, queue_size=1)
    publisher = rospy.Publisher('/lane_segmentation/image/compressed', CompressedImage)
    rospy.spin()
