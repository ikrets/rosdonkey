#!/usr/bin/env python

import rosbag
import os
import sys
from PIL import Image
from io import BytesIO
import numpy as np

if __name__ == '__main__':
    bag_file = sys.argv[1]

    folder = os.path.splitext(bag_file)[0]
    os.makedirs(folder)

    bag = rosbag.Bag(bag_file, 'r')

    try:
        for topic, msg, t in bag.read_messages(topics=['/raspicam_node/image/compressed',
                                                       '/lane_segmentation/image/compressed']):
            filename = lambda label, format: os.path.join(folder,
                                                          'frame_{:05d}_{}.{}'.format(msg.header.seq, label, format))

            if topic == '/raspicam_node/image/compressed':
                with open(filename('image', 'jpg'), 'wb') as fp:
                    fp.write(msg.data)

            if topic == '/lane_segmentation/image/compressed':
                with BytesIO(msg.data) as fp:
                    img = np.array(Image.open(fp))

                    # The lane predictions are stored in the G channel
                    Image.fromarray(img[:, :, 1] / 255).convert('L').save(filename('lanes', 'png'))

    except Exception as ex:
        print(ex)
        bag.close()