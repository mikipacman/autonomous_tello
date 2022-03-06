"""Module for detecting aruco markers"""
import collections
import os

import cv2
import numpy as np
import scipy.spatial.transform


class MarkerDetector:
    def __init__(
        self,
        camera_parameters,
        aruco_dictionary=cv2.aruco.DICT_4X4_50,
        aruco_marker_id_to_size={"default": 0.064},
        aruco_detector_params_corner_refinement_method=cv2.aruco.CORNER_REFINE_CONTOUR,
        queue_max_len=100,
    ):
        self.camera_parameters = camera_parameters
        self.aruco_dictionary = cv2.aruco.getPredefinedDictionary(aruco_dictionary)
        self.aruco_marker_id_to_size = aruco_marker_id_to_size
        self.aruco_detector_params = cv2.aruco.DetectorParameters_create()
        self.aruco_detector_params.cornerRefinementMethod = (
            aruco_detector_params_corner_refinement_method
        )

        self.marker_id_to_corners = {}
        self.queue_max_len = queue_max_len
        self.last_ids = None

    def add_image(self, img):
        corners, ids, _ = cv2.aruco.detectMarkers(
            img, self.aruco_dictionary, None, None, self.aruco_detector_params
        )

        if ids is not None:
            assert len(corners) == 1 and ids.shape[1] == 1, [corners, ids]
            assert len(ids.shape) == 2 and ids.shape[1] == 1, ids
            assert len(ids[:, 0]) == len(
                set(ids[:, 0])
            ), "There are some marker duplicates in the image!"

            ids = list(ids[:, 0])
            self.last_ids = ids
            corners = corners[0]

            # Update ids that are in the current img
            for c, i in zip(corners, ids):
                if i not in self.marker_id_to_corners:
                    self.marker_id_to_corners[i] = collections.deque(
                        maxlen=self.queue_max_len
                    )

                self.marker_id_to_corners[i].append(c)
        else:
            ids = []
            self.last_ids = []

        # Update ids that are NOT in the current img
        for i in set(self.marker_id_to_corners.keys()) - set(ids):
            if len(self.marker_id_to_corners[i]) == self.queue_max_len:
                self.marker_id_to_corners[i].pop()

    def detect_markers(self, averaged_over_n_frames=1):
        assert self.last_ids is not None, "detect_markers() called before add_image()"
        if len(self.last_ids) == 0:
            return None

        averaged_corners = [
            list(self.marker_id_to_corners[i])[-averaged_over_n_frames:]
            for i in self.last_ids
        ]
        averaged_corners = np.array(averaged_corners).mean(axis=1)
        corners = (averaged_corners,)

        # TODO support different aruco markers sizes at once
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
            corners,
            self.aruco_marker_id_to_size["default"],
            self.camera_parameters["camera_mat"],
            self.camera_parameters["dist_coeffs"],
        )
        return corners, np.array(self.last_ids)[..., np.newaxis], rvecs, tvecs


class ImageCalibrator:
    def __init__(
        self,
        path_to_images,
        chessboard_shape=(8, 5),
        camera_criteria=(
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            30,
            0.001,
        ),
    ):
        (
            self.dist_coeffs,
            self.mapx,
            self.mapy,
            self.camera_mat,
            self.roi,
        ) = self._get_calibration_parameters(
            path_to_images, chessboard_shape, camera_criteria
        )

    def undistort_img(self, img):
        dst = cv2.remap(img, self.mapx, self.mapy, cv2.INTER_LINEAR)
        x, y, w, h = self.roi
        dst = dst[y : y + h, x : x + w]
        return dst

    def get_camera_parameters(self):
        return {
            "dist_coeffs": self.dist_coeffs,
            "mapx": self.mapx,
            "mapy": self.mapy,
            "camera_mat": self.camera_mat,
            "roi": self.roi,
        }

    @staticmethod
    def _get_calibration_parameters(path_to_images, chessboard_shape, camera_criteria):
        paths = [os.path.join(path_to_images, i) for i in os.listdir(path_to_images)]
        cols, rows = chessboard_shape
        objp = np.zeros((cols * rows, 3), np.float32)
        objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
        objpoints = []
        imgpoints = []

        for img_path in paths:
            img = cv2.imread(img_path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, chessboard_shape, None)

            assert ret, img_path

            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(
                gray, corners, (11, 11), (-1, -1), camera_criteria
            )
            imgpoints.append(corners)

        ret, mtx, dist, _, _ = cv2.calibrateCamera(
            objpoints, imgpoints, gray.shape[::-1], None, None
        )
        h, w = img.shape[:2]
        new_camera_mat, roi = cv2.getOptimalNewCameraMatrix(
            mtx, dist, (w, h), 1, (w, h)
        )
        mapx, mapy = cv2.initUndistortRectifyMap(
            mtx, dist, None, new_camera_mat, (w, h), 5
        )

        return dist, mapx, mapy, new_camera_mat, roi


# Operations


def marker_coord_to_camera_coord(point, rvec, tvec):
    rot_mat = cv2.Rodrigues(rvec)[0]
    point = np.array(point)
    return rot_mat @ point + tvec


def rvec_to_angle(rvec):
    r = scipy.spatial.transform.Rotation.from_rotvec(rvec)
    return r.as_euler("xyz", degrees=True)[0][1]
