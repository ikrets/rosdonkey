import numpy as np
from PIL import Image


def linear_attract_repel_field(x,
                               attract_level=2,
                               repel_level=-10,
                               repel_width=10,
                               attract_value=26,
                               attract_width=14
                               ):
    result = np.zeros_like(x)

    result[(x > attract_value - attract_width / 2) & (x < attract_value + attract_width / 2)] = attract_level
    result[x < repel_width] = repel_level
    return result


def compute_path_on_fly(lane, f, ignore_border=20, steps=4):
    path = np.zeros_like(lane)

    y = path.shape[1] // 2
    x = 0

    prev_step = [steps, 0]
    max_it = 100

    lane = np.stack(np.where(lane), axis=1)
    distances_from_center = np.sqrt(np.linalg.norm(
        lane, axis=-1, ord=1
    ))[np.newaxis, :]

    going_back_steps = 0

    while y > ignore_border and y < path.shape[1] - ignore_border and x < path.shape[0] - ignore_border and max_it != 0:
        max_it -= 1
        path[x, y] = 1

        if prev_step[0] < 0:
            going_back_steps += 1
        else:
            going_back_steps = 0

        if going_back_steps > 3:
            break

        if not prev_step[0]:
            coords = np.array([[-steps, prev_step[1]], [0, prev_step[1]], [steps, prev_step[1]]])
        elif not prev_step[1]:
            coords = np.array([[prev_step[0], -steps], [prev_step[0], 0], [prev_step[0], steps]])
        else:
            coords = np.array([[prev_step[0], prev_step[1]], [prev_step[0], 0], [0, prev_step[1]]])
        coords += [x, y]

        distances = np.linalg.norm(coords[:, np.newaxis] - lane[np.newaxis, :], axis=-1, ord=2)
        field = np.sum(distances_from_center * f(distances), axis=-1)
        max_field_index = np.argmax(field)

        coords[max_field_index] = np.clip(coords[max_field_index], 0, None)
        prev_step = coords[max_field_index] - [x, y]
        x, y = coords[max_field_index]

    exit = 'stuck'

    if y <= ignore_border:
        exit = 'left'

    if y >= path.shape[1] - ignore_border:
        exit = 'right'

    if x >= path.shape[0] - ignore_border:
        exit = 'center'

    return path, exit


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
