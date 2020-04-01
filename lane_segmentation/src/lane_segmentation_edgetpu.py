#!/usr/bin/python3

import sys
# Needed to clear up PYTHONPATH from 2.7 python libraries
sys.path = [p for p in sys.path if '2.7' not in p]


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
        latency, result = self.run_inference(input_tensor)
        result = result.reshape((height, width, -1))

        return latency, result

if __name__ == '__main__':
    engine = SegmentationEngine(sys.argv[1])

    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind('tcp://127.0.0.1:{}'.format(sys.argv[2]))

    while True:
        img_bytes = socket.recv()
        img = np.frombuffer(img_bytes, np.uint8).reshape((64, 96, 3))

        latency, result = engine.segment(img)

        with BytesIO() as fp:
            result = np.squeeze(result > 0.5)
            fp.write(result.tobytes())
            fp.seek(0)
            socket.send(fp.read())
