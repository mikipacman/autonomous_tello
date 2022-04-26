"""Module for detecting aruco markers"""
import collections
import os

import cv2
import numpy as np
import scipy.spatial.transform

from lib.utils import assert_input_shapes, assert_output_shapes


class MarkerDetector:
    def __init__(
        self,
        camera_parameters,
        aruco_dictionary=cv2.aruco.DICT_4X4_50,
        aruco_marker_id_to_size={"default": 0.059},
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

        self.marker_id_to_stuff = collections.defaultdict(
            lambda: collections.deque(maxlen=queue_max_len)
        )
        self.last_ids = None

    def add_image(self, img):
        corners, ids, _ = cv2.aruco.detectMarkers(
            img, self.aruco_dictionary, None, None, self.aruco_detector_params
        )

        if ids is not None:
            if len(ids[:, 0]) != len(set(ids[:, 0])):
                print(
                    "There are some marker duplicates in the image! Skipping this frame"
                )
                return

            ids = list(ids[:, 0])
            self.last_ids = ids

            # TODO support different aruco markers sizes at once
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                corners,
                self.aruco_marker_id_to_size["default"],
                self.camera_parameters["camera_mat"],
                self.camera_parameters["dist_coeffs"],
            )

            # Update ids
            for i, c, rvec, tvec in zip(ids, corners, rvecs, tvecs):
                self.marker_id_to_stuff[i].append((c, rvec, tvec))
            for k in set(self.marker_id_to_stuff.keys()) - set(ids):
                self.marker_id_to_stuff[k].append(None)
        else:
            self.last_ids = None

    def detect_markers(self, averaged_over_n_frames=1):
        if not self.last_ids:
            return None
        # Get a slice
        averaged_stuff = [
            list(self.marker_id_to_stuff[i])[-averaged_over_n_frames:]
            for i in self.last_ids
        ]

        # Remove None values
        averaged_stuff = [[c for c in l if c is not None] for l in averaged_stuff]

        # Average corners
        corners = [[c[0] for c in l] for l in averaged_stuff]
        rvecs = [[c[1] for c in l] for l in averaged_stuff]
        tvecs = [[c[2] for c in l] for l in averaged_stuff]

        # Take last
        corners = tuple(np.array(l)[-1] for l in corners)
        tvecs = tuple(np.array(l)[-1] for l in tvecs)

        # Take median
        rvecs = tuple(np.median(np.array(l), axis=0) for l in rvecs)

        return (
            corners,
            np.array(self.last_ids)[..., np.newaxis],
            np.array(rvecs),
            np.array(tvecs),
        )


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


CoordinateChange = collections.namedtuple(
    "CoordinateChange", ["translation", "rotation"]
)


class CoordinateTranslator:
    def __init__(self, marker_id_to_coordinate_change, camera_parameters):
        self.marker_id_to_coordinate_change = marker_id_to_coordinate_change
        for k, v in self.marker_id_to_coordinate_change.items():
            self.marker_id_to_coordinate_change[k] = CoordinateChange(
                rotation=v.rotation,
                translation=np.array(v.translation, dtype=np.float32),
            )
        self.camera_parameters = camera_parameters

    @assert_input_shapes(None, (6, 3), (1, 3), (1, 3))
    @assert_output_shapes((6, 3))
    def marker_to_camera(self, points, rvec, tvec):
        rvec = np.array(rvec)
        tvec = np.array(tvec)
        points = np.array(points)
        rot_mat = scipy.spatial.transform.Rotation.from_rotvec(rvec).as_matrix()[0]
        return (rot_mat @ points.T + tvec.T).T

    @assert_input_shapes(None, (6, 3), None)
    @assert_output_shapes((6, 3))
    def global_to_marker(self, points, target_marker_id):
        coordinate_change = self.marker_id_to_coordinate_change[target_marker_id]
        rot_mat = coordinate_change.rotation.inv().as_matrix()
        a = (rot_mat @ (points - coordinate_change.translation).T).T
        return a

    @assert_input_shapes(None, (6, 3))
    @assert_output_shapes((6, 3))
    def camera_to_tello(self, points):
        points = np.array(points) * 100
        return [(int(p[2]), int(-p[0]), int(-p[1])) for p in points]

    def rvec_to_angle(self, rvec):
        r = scipy.spatial.transform.Rotation.from_rotvec(rvec)
        return r.as_euler("xyz", degrees=True)[0][1]

    @assert_input_shapes(None, (6, 3))
    @assert_output_shapes((6, 2))
    def camera_to_img(self, points):
        points_in_img = self.camera_parameters["camera_mat"] @ points.T
        points_in_img /= points_in_img[2]
        points_in_img = points_in_img[:2, :].astype(np.int64).T
        return points_in_img
