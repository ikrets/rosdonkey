import numpy as np
import math
from time import time
from PIL import Image
import cv2

downsample = 1

all_points = np.array([(x, y) for x in range(0, 32, downsample) for y in range(0, 48, downsample)])
all_points = all_points.reshape((-1, 2))


def f(x):
    scale = 3
    y = np.empty_like(x)
    y[x < 20] = 1.5 * np.tanh(x[x < 20] / scale - 4) - .5
    y[x >= 20] = np.tanh(((40 - x[x >= 20]) / scale - 4)) / 2 + 0.5
    return y


def compute_field(img, variance):
    lane_points = np.stack(np.where(img), axis=1)
    distances = lane_points[np.newaxis, :] - all_points[:, np.newaxis]
    distances = np.linalg.norm(distances, axis=-1, ord=1)
    point_fields = f(distances)
    field = np.sum(point_fields, axis=1).reshape((math.ceil(29 / downsample), -1))

    field = np.array(Image.fromarray(field).resize((img.shape[1], img.shape[0]), Image.BILINEAR))

    return field


def compute_path_on_fly(lane):
    path = np.zeros_like(lane)
    x = 0
    y = path.shape[1] // 2
    prev_step = [1, 0]
    max_it = 1000

    lane = np.stack(np.where(lane), axis=1)

    while y > 3 and y < path.shape[1] - 1 and x < path.shape[0] - 1 and max_it != 0:
        max_it -= 1
        path[x, y] = 1

        if not prev_step[0]:
            coords = np.array([[1, 0], [0, prev_step[1]], [1, prev_step[1]]])
        elif not prev_step[1]:
            coords = np.array([[prev_step[0], -1], [prev_step[0], 0], [prev_step[0], 1]])
        else:
            coords = np.array([[prev_step[0], prev_step[1]], [prev_step[0], 0], [0, prev_step[1]]])
        coords = coords[coords[:, 0] >= 0]
        coords += [x, y]

        coords = coords[np.logical_and(coords[:, 0] > 0, coords[:, 1] > 0)]

        max_field = -np.inf
        for i in range(len(coords)):
            field = 0
            distances = np.linalg.norm(coords[i] - lane, axis=-1, ord=1)
            field += np.sum(f(distances))

            field += 5 * np.linalg.norm(coords[i] - [0, path.shape[1] // 2])

            if field > max_field:
                max_field = field
                max_field_index = i

        coords[max_field_index] = np.clip(coords[max_field_index], 0, None)
        prev_step = coords[max_field_index] - [x, y]
        x, y = coords[max_field_index] + prev_step

    return path


def progress_field(img):
    initial_point = np.array([0, img.shape[1] // 2])
    distances = initial_point[np.newaxis, :] - all_points
    distances = np.linalg.norm(distances, axis=-1, ord=1) ** 2
    field = distances.reshape((math.ceil(29 / downsample), -1))
    field = np.array(Image.fromarray(field).resize((img.shape[1], img.shape[0]), Image.BILINEAR))

    return field


def compute_path(field):
    path = np.zeros_like(field)
    x = 0
    y = path.shape[1] // 2
    prev_step = [1, 0]
    max_it = 1000

    while y > 3 and y < path.shape[1] - 1 and x < path.shape[0] - 1 and max_it != 0:
        max_it -= 1
        path[x, y] = 1

        if not prev_step[0]:
            coords = np.array([[1, 0], [0, prev_step[1]], [1, prev_step[1]]])
        elif not prev_step[1]:
            coords = np.array([[prev_step[0], -1], [prev_step[0], 0], [prev_step[0], 1]])
        else:
            coords = np.array([[prev_step[0], prev_step[1]], [prev_step[0], 0], [0, prev_step[1]]])
        coords = coords[coords[:, 0] >= 0]
        coords += [x, y]

        coords = coords[np.logical_and(coords[:, 0] > 0, coords[:, 1] > 0)]
        m = np.argmax(field[coords[:, 0], coords[:, 1]])
        coords[m] = np.clip(coords[m], 0, None)
        prev_step = coords[m] - [x, y]
        x, y = coords[m]

    return path

def compute_path_polynom(lane_mask):
    alpha = 0.1
    lane_field = compute_field(lane_mask, 100)
    progress = progress_field(lane_mask)

    lane_field /= np.mean(lane_field)
    progress /= np.mean(progress)

    field = alpha * lane_field + (1 - alpha) * progress
    path = compute_path(field)

    x, y = np.where(path)
    w = np.ones_like(x)
    w[0] = 10000

    return np.polyfit(x, y, deg=2, w=w)

def compute_polynom(path):
    x, y = np.where(path)
    w = np.ones_like(x)
    w[0] = 10000

    return np.polyfit(x, y, deg=2, w=w)


def compute_path_on_fly(lane):
    path = np.zeros_like(lane)
    x = 0
    y = path.shape[1] // 2
    prev_step = [1, 0]
    max_it = 1000

    lane = np.stack(np.where(lane), axis=1)

    while y > 3 and y < path.shape[1] - 1 and x < path.shape[0] - 1 and max_it != 0:
        max_it -= 1
        path[x, y] = 1

        if not prev_step[0]:
            coords = np.array([[1, 0], [0, prev_step[1]], [1, prev_step[1]]])
        elif not prev_step[1]:
            coords = np.array([[prev_step[0], -1], [prev_step[0], 0], [prev_step[0], 1]])
        else:
            coords = np.array([[prev_step[0], prev_step[1]], [prev_step[0], 0], [0, prev_step[1]]])
        coords = coords[coords[:, 0] >= 0]
        coords += [x, y]

        coords = coords[np.logical_and(coords[:, 0] > 0, coords[:, 1] > 0)]

        max_field = -np.inf
        for i in range(len(coords)):
            field = 0
            distances = np.linalg.norm(coords[i] - lane, axis=-1, ord=1)
            field += np.sum(f(distances))

            field += 5 * np.linalg.norm(coords[i] - [0, path.shape[1] // 2])

            if field > max_field:
                max_field = field
                max_field_index = i

        coords[max_field_index] = np.clip(coords[max_field_index], 0, None)
        prev_step = coords[max_field_index] - [x, y]
        x, y = coords[max_field_index] + 2 * prev_step

    return path


if __name__ == '__main__':
    from sys import argv
    from timeit import timeit

    filename = argv[1]
    img = np.array(Image.open(filename))
    img = transform(img)

    times = 25
    print(timeit(lambda: compute_path_on_fly(img), number=times) / times)
    path = compute_path_on_fly(img)
    print(timeit(lambda: compute_polynom(path), number=times) / times)
