#!/usr/bin/python3

import sys
# Needed to clear up PYTHONPATH from 2.7 python libraries
sys.path = [p for p in sys.path if '2.7' not in p]

import numpy as np
from PIL import Image
from io import BytesIO
from edgetpu.basic.basic_engine import BasicEngine
import zmq
from transform import undistort_birdeyeview

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
    engine = SegmentationEngine(sys.argv[1])

    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind('tcp://127.0.0.1:{}'.format(sys.argv[2]))

    while True:
        img_bytes = socket.recv()

        img = np.frombuffer(img_bytes, dtype=np.uint8)
        img = img.reshape(480, 640, 3).copy()

        # The camera has a weird white line on the right edge.
        img[:, -2:, :] = 0

        img = undistort_birdeyeview(img[-256:, :, :])

        _, result = engine.segment(img)

        with BytesIO() as fp:
            result = np.squeeze(result)
            img = Image.fromarray(result * 255).convert('RGB')
            img.save(fp, format='PNG')
            fp.seek(0)
            socket.send(fp.read())
