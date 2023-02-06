#!/usr/bin/python3

import sys
# Needed to clear up PYTHONPATH from 2.7 python libraries
sys.path = [p for p in sys.path if '2.7' not in p]

# TODO fast hacked for now
sys.path.append('/home/ubuntu/rosdonkey/src/dataset_utils')
from transform import make_undistort_birdeye

import numpy as np
from PIL import Image
from io import BytesIO
from edgetpu.basic.basic_engine import BasicEngine
import zmq
import cv2
from time import time

class SegmentationEngine(BasicEngine):
    def __init__(self, model_path):
        BasicEngine.__init__(self, model_path)

    def segment(self, img):
        input_tensor_shape = self.get_input_tensor_shape()
        _, height, width, _ = input_tensor_shape

        input_tensor = np.asarray(img).flatten()
        latency, result = self.RunInference(input_tensor)
        result = result.reshape((height, width, -1))

        return latency, result

if __name__ == '__main__':
    undistort_birdeyeview = make_undistort_birdeye(
        input_shape=(320, 240),
        target_shape=(32, 48))

    engine = SegmentationEngine(sys.argv[1])

    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind('tcp://127.0.0.1:{}'.format(sys.argv[2]))

    while True:
        img_bytes = socket.recv()

        img = np.frombuffer(img_bytes, dtype=np.uint8)
        img = img.reshape(240, 320, 3).copy()

        # The camera has a weird white line on the right edge.
        img[:, -2:, :] = 0
        img = undistort_birdeyeview(img)
        latency, result = engine.segment(img)

        with BytesIO() as fp:
            result = np.squeeze(result > 0.5)
            fp.write(result.tobytes())
            fp.seek(0)
            socket.send(fp.read())
