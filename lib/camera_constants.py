import cv2

# Settings for calibrating camera
camera_chessboard_shape = (8, 5)
camera_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Aruco detector parameters
aruco_dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
aruco_detector_params = cv2.aruco.DetectorParameters_create()
aruco_detector_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_CONTOUR
aruco_marker_side = 0.064
