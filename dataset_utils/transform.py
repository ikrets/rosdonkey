import numpy as np
import cv2

def make_undistort_birdeye(input_shape, target_shape):
    DIM = (640, 480)
    K = np.array(
        [[309.3663656629932, 0.0, 303.7312979054133], [0.0, 307.6446025077482, 225.6589032486636], [0.0, 0.0, 1.0]])
    D = np.array([[-0.02454023737239999], [-0.05241193093993287], [0.068734121741902], [-0.03237837863118788]])

    src = np.array([[208, 376], [278, 274], [337, 298], [323, 378]], dtype=np.float32)

    # 0, 0 is center of the car, line after wheels end
    # y forward x to the right
    dst = np.array([[-9, 8], [-5, 28], [5, 20], [2, 8]], dtype=np.float32)

    src *= float(input_shape[0]) / DIM[0]
    dst *= float(target_shape[1]) / 96
    desired_size = target_shape
    dst[:, 0] += float(desired_size[1]) / 2

    K *= float(input_shape[0]) / DIM[0]
    K[2, 2] = 1.

    nK = K.copy()
    nK[0, 0] /= 2
    nK[1, 1] /= 2

    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3),
                                                     nK, input_shape, cv2.CV_16SC2)
    undistort = lambda img, dst: cv2.remap(img, map1, map2, 
        interpolation=cv2.INTER_NEAREST,
        dst=dst)

    perspective_transform = cv2.getPerspectiveTransform(src, dst)
    iTM = cv2.invert(perspective_transform)[1]
    trans_map_x = np.empty(dtype=np.float32, shape=desired_size)
    trans_map_y = np.empty(dtype=np.float32, shape=desired_size)

    for y in range(trans_map_x.shape[0]):
        fy = float(y)
        for x in range(trans_map_y.shape[1]):
            fx = float(x)
            w = iTM[2, 0] * fx + iTM[2, 1] * fy + iTM[2, 2]
            w = 1. / w if w != 0. else 0.

            trans_map_x[y, x] = (iTM[0, 0] * fx + iTM[0, 1] * fy + iTM[0, 2]) * w
            trans_map_y[y, x] = (iTM[1, 0] * fx + iTM[1, 1] * fy + iTM[1, 2]) * w

    trans_map_x, trans_map_y = cv2.convertMaps(trans_map_x, trans_map_y, cv2.CV_16SC2)
    birdeye = lambda img, dst: cv2.remap(img, trans_map_x, trans_map_y, 
        interpolation=cv2.INTER_LINEAR,
        dst=dst)

    def undistort_birdeye(img, dst):
        undistorted_img = np.empty((input_shape[1], input_shape[0], 3), dtype=np.uint8)
        undistort(img, undistorted_img)
        birdeye(undistorted_img, dst)

    return undistort_birdeye

if __name__ == '__main__':
    from PIL import Image
    import cv2

    img = cv2.imread('/home/ilya/donkey_data/drivery_18_07_no_lights_ss10000/frame0494.jpg')
    img = cv2.resize(img, (320, 240), cv2.INTER_LINEAR)
    cv2.imshow('image', img)
    transform = make_undistort_birdeye((320, 240), (32, 48))
    cv2.imshow('transformed', transform(img))
    cv2.waitKey()
    cv2.destroyAllWindows()
