#!/usr/bin/env python

import rosbag
import os
import sys

if __name__ == '__main__':
    bag_file = sys.argv[1]

    folder = os.path.splitext(bag_file)[0]
    os.makedirs(folder)

    bag = rosbag.Bag(bag_file, 'r')

    try:
        for topic, msg, t in bag.read_messages(topics=['/raspicam_node/image/compressed']):
            img_filename = 'frame_{:05d}.{}'.format(msg.header.seq, msg.format)
            with open(os.path.join(folder, img_filename), 'wb') as fp:
                fp.write(msg.data)
    except Exception as ex:
        print(ex)
        bag.close()