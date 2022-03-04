import collections
import os
import time
from subprocess import DEVNULL, check_call

import cv2
import numpy as np
import scipy.spatial.transform
from termcolor import cprint

from lib.camera_constants import (aruco_detector_params, aruco_dictionary,
                                  aruco_marker_side, camera_chessboard_shape,
                                  camera_criteria)


# Wifi
def connect_to_wifi(name, password, num_retries=20):
    cmd_refresh_list = "sudo iwlist wlp4s0 scan"
    cmd_connect = "nmcli d wifi connect {} password {}".format(name, password)

    for i in range(num_retries):
        try:
            check_call(cmd_refresh_list.split(), stdout=DEVNULL, stderr=DEVNULL)
            check_call(cmd_connect.split(), stdout=DEVNULL, stderr=DEVNULL)
        except:
            cprint(
                f"\rCould not connect to {name}. Retrying {i + 1}/{num_retries} ...",
                "yellow",
                end="",
            )
            continue
        if i:
            print()
        return True

    print()
    return False


def connect_or_exit(server_name, password):
    if not connect_to_wifi(server_name, password):
        cprint(f"Failed to connect to wifi {server_name}", "red")
        exit(1)
    else:
        cprint(f"Connected to {server_name}", "green")


# Fps
class FPS:
    def __init__(self, avarageof=200):
        self.frametimestamps = collections.deque(maxlen=avarageof)

    def __call__(self):
        self.frametimestamps.append(time.time())
        if len(self.frametimestamps) > 1:
            return len(self.frametimestamps) / (
                self.frametimestamps[-1] - self.frametimestamps[0]
            )
        else:
            return 0.0


# cv2 drawing utils
color_to_rgb = {
    "white": (255, 255, 255),
    "red": (0, 0, 255),
    "blue": (255, 0, 0),
    "black": (0, 0, 0),
    "green": (0, 255, 0),
}


def draw_text(img, text, pos, color="white"):
    return cv2.putText(
        img,
        text,
        pos,
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        color_to_rgb[color],
        2,
        cv2.LINE_AA,
    )


# Camera matrix
def get_calibration_parameters(path_to_images):
    paths = [os.path.join(path_to_images, i) for i in os.listdir(path_to_images)]
    cols, rows = camera_chessboard_shape
    objp = np.zeros((cols * rows, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane

    for img_path in paths:
        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, camera_chessboard_shape, None)

        assert ret, img_path

        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), camera_criteria)
        imgpoints.append(corners)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None
    )
    h, w = img.shape[:2]
    new_camera_mat, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, new_camera_mat, (w, h), 5)
    return ret, mtx, dist, rvecs, tvecs, mapx, mapy, new_camera_mat, roi


def get_marker_from_image(img, params, marker_dict):
    # TODO this is to specific, move it to tello morelo class or make a new class for managing markers
    # Make a "MarkerDetector" class and store ~500 recent marker corners, then give a method for estimateing
    # pose from n most recent corners, that will increase the accuract of the estimate.
    # Also move all cv2 params to this class and add support for markers with different sizes
    corners, ids, _ = cv2.aruco.detectMarkers(
        img, aruco_dictionary, None, None, aruco_detector_params
    )
    if ids is None:
        return None
    else:
        assert len(ids.shape) == 2 and ids.shape[1] == 1, ids
        assert len(ids[:, 0]) == len(
            set(ids[:, 0])
        ), "There are some marker duplicates in the image!"
        _, _, dist, _, _, _, _, new_camera_mat, _ = params

        corners = average_corners(corners, ids, marker_dict)

        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
            corners, aruco_marker_side, new_camera_mat, dist
        )
        return corners, ids, rvecs, tvecs


def average_corners(corners, ids, marker_dict):
    assert len(corners) == 1 and ids.shape[1] == 1, [corners, ids]
    ids = list(ids[:, 0])
    corners = corners[0]

    for c, i in zip(corners, ids):
        if i not in marker_dict:
            marker_dict[i] = collections.deque(maxlen=30)

        marker_dict[i].append(c)

    averaged_corners = [list(marker_dict[i]) for i in ids]
    averaged_corners = np.array(averaged_corners)
    averaged_corners = averaged_corners.mean(axis=1)

    return (averaged_corners,)


def undistort_img(img, params):
    _, _, _, _, _, mapx, mapy, _, roi = params
    dst = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)
    x, y, w, h = roi
    dst = dst[y : y + h, x : x + w]
    return dst


def marker_coord_to_camera_coord(point, rvec, tvec):
    rot_mat = cv2.Rodrigues(rvec)[0]
    point = np.array(point)
    return rot_mat @ point + tvec


def rvec_to_angle(rvec):
    r = scipy.spatial.transform.Rotation.from_rotvec(rvec)
    return r.as_euler("xyz", degrees=True)[0][1]
