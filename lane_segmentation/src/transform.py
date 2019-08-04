import numpy as np
import cv2

camera_matrix = np.array(
    [1266.78359602357, 0, 608.4607470200075, 0, 1267.334024187105, 374.6054937400205, 0, 0, 1]).reshape((3, 3))
distortion_coeffs = np.array([0.06107748316798726, -0.2708303958945525, 0.002115970398267557, -0.008895203586217777, 0])

rescale = np.ones((3, 3))
rescale[0, :] *= 0.5
rescale[1, :] *= 480.0 / 720.0

camera_matrix *= rescale
new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, distortion_coeffs, (640, 480), 1)

roi = (2, 0.6)
scale = 50 * 1.6 * 96 / 160

src = np.array([[237, 362], [368, 367], [274, 267], [344, 270]], dtype=np.float32)
src[:, 1] -= 480 - 256
dst = np.array([[0, 0], [.12, 0], [0, .16], [.12, .16]], dtype=np.float32)
dst[:, 0] += roi[0] / 2 - 0.06
dst[:, 1] += 0.1

dst *= scale
perspective_transform = cv2.getPerspectiveTransform(src, dst)

undistort_map_x, undistort_map_y = cv2.initUndistortRectifyMap(
    camera_matrix, distortion_coeffs, np.eye(3), new_camera_matrix, (640, 256),
    cv2.CV_16SC2)

iTM = cv2.invert(perspective_transform)[1]
trans_map_x = np.empty(dtype=np.float32, shape=(29, 93))
trans_map_y = np.empty(dtype=np.float32, shape=(29, 93))

for y in range(trans_map_x.shape[0]):
    fy = float(y)
    for x in range(trans_map_y.shape[1]):
        fx = float(x)
        w = iTM[2, 0] * fx + iTM[2, 1] * fy + iTM[2, 2]
        w = 1. / w if w != 0. else 0.
        
        trans_map_x[y, x] = (iTM[0, 0] * fx + iTM[0, 1] * fy + iTM[0, 2]) * w
        trans_map_y[y, x] = (iTM[1, 0] * fx + iTM[1, 1] * fy + iTM[1, 2]) * w
        
trans_map_x, trans_map_y = cv2.convertMaps(trans_map_x, trans_map_y, cv2.CV_16SC2)
        
def undistort_birdeyeview(img):
    if len(img.shape) == 2:
        img = img[:, :, np.newaxis]

    undistort_img = cv2.remap(img, undistort_map_x, undistort_map_y,
                             interpolation=cv2.INTER_NEAREST)
    return cv2.remap(undistort_img, trans_map_x, trans_map_y,
                          interpolation=cv2.INTER_NEAREST)
