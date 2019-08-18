import numpy as np
import requests
from PIL import Image
import os
from skimage.draw import polygon
import cv2
from io import BytesIO

import argparse

parser = argparse.ArgumentParser(description='Extract labeled images from the semantic segmentation editor.')
parser.add_argument('images-folder', type=str,
                    help='same as images-folder in the semantic segmentation editor')
parser.add_argument('output-folder', type=str,
                    help='folder to write results into')

args = vars(parser.parse_args())

base_dir = args['images-folder']
output_dir = args['output-folder']

crop = (0, 480 - 256, 640, 480)

examples = []
class_appearances = [0, 0]

for n in ['images', 'masks', 'images_with_masks']:
    os.makedirs(os.path.join(output_dir, n), exist_ok=True)

for item in requests.get('http://localhost:3000/api/listing').json():
    item['folder'] = item['folder'][1:]

    filename = os.path.join(base_dir, item['folder'], item['file'])
    if not os.path.exists(filename):
        print('Missing {}'.format(filename))
        continue

    img = Image.open(filename)
    original_img_size = img.size
    assert(640 % img.size[0] == 0)
    assert(480 % img.size[1] == 0)
    assert(img.size[0] / 640 == img.size[1] / 480)

    adjusted_crop = tuple(int(c * img.size[0] / 640) for c in crop)

    img = img.crop(adjusted_crop)
    image_bytes = BytesIO()
    img.save(image_bytes, format='PNG')
    output_filename = '{}_{}'.format(item['folder'].replace('/', '-'), item['file'])

    labels = requests.get('http://localhost:3000/api/json' + item['url']).json()
    mask = np.zeros(original_img_size[::-1], dtype=np.uint8)

    class1 = 0
    for label in labels['objects']:
        xs = []
        ys = []

        for point in label['polygon']:
            xs.append(point['x'])
            ys.append(point['y'])

        p = polygon(xs, ys)
        p_inside_img = (p[0] >= 0) & (p[0] < original_img_size[0]) & (p[1] >= 0) & (p[1] < original_img_size[1])
        mask[p[1][p_inside_img], p[0][p_inside_img]] = 1

    mask = mask[adjusted_crop[1]:adjusted_crop[3], adjusted_crop[0]:adjusted_crop[2]]
    _, mask_bytes = cv2.imencode('.png', mask)

    img.save(os.path.join(output_dir, 'images', output_filename))

    alpha = 0.5
    overlayed = np.array(img)
    overlayed[mask == 1] = (alpha * np.array([0, 255, 0]) + (1 - alpha) * overlayed[mask == 1]).round()
    cv2.imwrite(os.path.join(output_dir, 'images_with_masks', output_filename), overlayed)
    cv2.imwrite(os.path.join(output_dir, 'masks', os.path.splitext(output_filename)[0] + '.png'), mask)
