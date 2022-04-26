import logging
import time

import cv2
import djitellopy
import numpy as np
from termcolor import cprint

from lib.image_processing import (CoordinateTranslator, ImageCalibrator,
                                  MarkerDetector)
from lib.utils import FPS, color_to_rgb, connect_or_exit, draw_text

# Set logging level
djitellopy.Tello.LOGGER.setLevel(logging.WARNING)

help_prompt = """
Steering:
  w       i
a s d   j k l

Emergency shutdown:     spacebar
Takeoff/Land:           e
Velocity:               -/+
Averaged over n frames: shift and -/+
Toggle state window:    1
Toggle draw axis:       2
Print help:             h
Quit:                   q
"""


class Tello:
    def __init__(
        self,
        path_to_calibration_images,
        display_points_in_marker_coord,
        display_points_in_global_coord,
        marker_id_to_coordinate_change,
        init_velocity=50,
        wifi_name="TELLO-MORELO",
        password="twojastara",
    ):
        # Connect to tello
        connect_or_exit(wifi_name, password)

        # Tello init
        self.tello = djitellopy.Tello()
        self.tello.connect()
        self.tello.streamoff()
        self.tello.streamon()

        # Things for moving
        self.velocity = init_velocity
        self.left_right_velocity = 0
        self.for_back_velocity = 0
        self.up_down_velocity = 0
        self.yaw_velocity = 0
        self.is_flying = False

        # CV2 windows
        self.cv2_camera_window_name = "tello"
        self.cv2_state_window_name = "state"
        self.state_window_active = False

        # Things for plotting current image
        self.fps_counter = FPS()
        self.frame_count = 0
        self.draw_marker_axes = True
        self.display_points_in_marker_coord = display_points_in_marker_coord
        self.display_points_in_global_coord = display_points_in_global_coord

        # Things for plotting state data
        self.rpy_names = ["roll", "pitch", "yaw"]
        self.v_names = ["vgx", "vgy", "vgz"]
        self.a_names = ["agx", "agy", "agz"]
        self.h_names = ["tof", "h", "baro"]
        self.other_names = ["templ", "temph", "bat", "time"]

        # Image processing
        self.image_calibrator = ImageCalibrator(path_to_calibration_images)
        self.marker_detector = MarkerDetector(
            camera_parameters=self.image_calibrator.get_camera_parameters()
        )
        self.averaged_over_n_frames = 3
        self.coordinate_translator = CoordinateTranslator(
            marker_id_to_coordinate_change=marker_id_to_coordinate_change.id_to_marker_map,
            camera_parameters=self.image_calibrator.get_camera_parameters(),
        )

        # TODO this does not work, can we fix it?
        # self.tello.set_video_direction(djitellopy.Tello.CAMERA_FORWARD)
        # self.tello.set_video_bitrate(djitellopy.Tello.BITRATE_1MBPS)
        # self.tello.set_video_fps(djitellopy.Tello.FPS_5)

        # TODO add an option for plotting stats during manouvers

    def get_data(self, display=True, interactive=False):
        """Main function for reading drone's data. It also shows images seen by a drone
        and allows user to control the drone using keyboard"""

        frame_read = self.tello.get_frame_read()
        state = self.tello.get_current_state()
        self.frame_count += 1
        img = self.image_calibrator.undistort_img(frame_read.frame.copy())
        self.marker_detector.add_image(img)

        if display:
            key = cv2.pollKey()
            img = self._img_to_display(img, key, state)
            cv2.imshow(self.cv2_camera_window_name, img)
            cv2.moveWindow(self.cv2_camera_window_name, 0, 0)

            if self.state_window_active:
                img_state = self._state_to_display(state)
                cv2.imshow(self.cv2_state_window_name, img_state)
                cv2.moveWindow(self.cv2_state_window_name, 1200, 0)

            if interactive:
                self.handle_key_commands(key)

            if key == ord(" "):
                self.emergency()

        return frame_read.frame, state

    def takeoff(self):
        """Takeoff. Active wating allows for live view during Taking off"""

        if not self.is_flying:
            self.tello.send_command_without_return("takeoff")
            self.is_flying = True
            self.sleep(7)

    def land(self):
        """Land. Active wating allows for live view during landing"""

        if self.is_flying:
            self.tello.send_command_without_return("land")
            self.is_flying = False
            self.sleep(7)

    def emergency(self):
        """Shutdown all motors"""
        self.tello.emergency()
        cprint("EMERGENCY SHUTDOWN!", "red")
        self.__del__()

    # TODO add sleep until tello finisheds manouver (its velocities are 0)
    def sleep(self, sec, interactive=False):
        """Actively sleep. It allows viewing drone's view while sleeping"""

        start_time = time.time()
        while time.time() - start_time < sec:
            self.get_data(interactive=interactive)

    def find_any_markers(self, marker_ids, rotation):
        """Rotate until you find any of given markers, then center it and return its ID"""

        # TODO consider other variants such as "centering marker" and not just aligining to it
        # TODO make the movement cointinous by using send_rc_control(yaw_speed)
        # TODO print stats after completing the manouver (current angle, (x, y, z) etc)

        assert rotation in ("clockwise", "anticlockwise")
        cmd = "cw" if rotation == "clockwise" else "ccw"

        while True:
            self.get_data()
            ret = self.marker_detector.detect_markers(averaged_over_n_frames=10)
            if ret:
                _, ids, rvecs, _ = ret
                found = [m in ids for m in marker_ids]
                if any(found):
                    marker_idx = marker_ids[found.index(True)]
                    angle = self.coordinate_translator.rvec_to_angle(rvecs[marker_idx])
                    cmd = "cw" if angle > 0 else "ccw"
                    self.tello.send_command_with_return(f"{cmd} {int(np.abs(angle))}")
                    self.sleep(2)
                    return marker_idx

            self.tello.send_command_without_return(f"{cmd} 20")
            self.sleep(0.5)

    def fly_to_point_in_camera_coord(self, point, velocity=None):
        """Fly to a point in camera coordinates system"""

        if not velocity:
            velocity = self.velocity

        point_in_tello_coord = self.coordinate_translator.camera_to_tello(point)
        self.tello.go_xyz_speed(*point_in_tello_coord, velocity)
        self.sleep(5)

    def fly_to_point_in_marker_coord(
        self,
        point,
        marker_id,
        velocity=None,
        num_tries=10,
        position_averaged_over_n_frames=10,
    ):
        """Fly to a point in marker coordinates system"""

        # TODO plot stats before and after manouer (xyz, distance from point etc)
        # TODO use data["xspeed"] or tello.get_velocity to know when to the manouver is finished

        for _ in range(position_averaged_over_n_frames):
            self.get_data()

        for _ in range(num_tries):
            ret = self.marker_detector.detect_markers(
                averaged_over_n_frames=position_averaged_over_n_frames
            )
            if ret:
                _, ids, rvecs, tvecs = ret
                assert marker_id in ids
                marker_idx = list(ids).index(marker_id)
                camera_point = self.coordinate_translator.marker_to_camera(
                    point, rvecs[marker_idx], tvecs[marker_idx]
                )
                self.fly_to_point_in_camera_coord(camera_point, velocity)
                return
        raise Exception(f"Marker {marker_id} not found!")

    def handle_key_commands(self, key):
        """Handle key captured by cv2 window and send commands to the drone"""

        self.yaw_velocity = 0
        self.up_down_velocity = 0
        self.for_back_velocity = 0
        self.left_right_velocity = 0

        # Up down and angle
        if key == ord("i"):
            self.for_back_velocity += self.velocity
        elif key == ord("k"):
            self.for_back_velocity -= self.velocity
        elif key == ord("j"):
            self.left_right_velocity -= self.velocity
        elif key == ord("l"):
            self.left_right_velocity += self.velocity

        # For back left right
        elif key == ord("w"):
            self.up_down_velocity += self.velocity
        elif key == ord("s"):
            self.up_down_velocity -= self.velocity
        elif key == ord("a"):
            self.yaw_velocity -= self.velocity
        elif key == ord("d"):
            self.yaw_velocity += self.velocity

        # Additional steering
        elif key == ord("e"):
            if not self.is_flying:
                self.takeoff()
            else:
                self.land()
        elif key == ord("-"):
            self.velocity = max(10, self.velocity - 1)
        elif key == ord("="):
            self.velocity = min(100, self.velocity + 1)
        elif key == ord("_"):
            self.averaged_over_n_frames = max(1, self.averaged_over_n_frames - 1)
        elif key == ord("+"):
            self.averaged_over_n_frames += 1

        # Change display
        elif key == ord("1"):
            self.state_window_active = not self.state_window_active
            if not self.state_window_active:
                cv2.destroyWindow(self.cv2_state_window_name)
        elif key == ord("2"):
            self.draw_marker_axes = not self.draw_marker_axes

        # Quit
        elif key == ord("q"):
            self.__del__()

        # Help
        elif key == ord("h"):
            print(help_prompt)

        self.tello.send_rc_control(
            self.left_right_velocity,
            self.for_back_velocity,
            self.up_down_velocity,
            self.yaw_velocity,
        )

    def _img_to_display(self, img, key, state):
        # Constant part to display
        img = draw_text(img, f"FPS={int(self.fps_counter())}", (10, 30))
        img = draw_text(img, f"anf={int(self.averaged_over_n_frames)}", (10, 60))
        img = draw_text(img, f"key={key}", (10, 90))
        img = draw_text(img, f"vel={self.velocity}", (10, 120))
        img = draw_text(img, f"time={state['time']}", (10, 150))

        # Blinking battery
        bat_color = (
            "red" if state["bat"] < 20 and self.frame_count // 30 % 2 == 0 else "white"
        )
        img = draw_text(img, f"bat={state['bat']}", (10, 180), bat_color)

        # Blinking temp
        temp = (state["temph"] + state["templ"]) // 2
        temp_color = "red" if temp > 90 and self.frame_count // 30 % 2 == 0 else "white"
        img = draw_text(img, f"temp={temp}", (10, 210), temp_color)

        # Draw markers
        if self.draw_marker_axes:
            ret = self.marker_detector.detect_markers(
                averaged_over_n_frames=self.averaged_over_n_frames
            )
            if ret:
                corners, ids, rvecs, tvecs = ret
                cv2.aruco.drawDetectedMarkers(img, corners, ids)
                camera_params = self.image_calibrator.get_camera_parameters()

                for rvec, tvec, marker_id in zip(rvecs, tvecs, ids):
                    marker_id = marker_id[0]
                    cv2.aruco.drawAxis(
                        img,
                        camera_params["camera_mat"],
                        camera_params["dist_coeffs"],
                        rvec,
                        tvec,
                        0.1,
                    )

                    # Draw points in marker coordinates
                    if marker_id in self.display_points_in_marker_coord:
                        for point in self.display_points_in_marker_coord[marker_id]:
                            points = self._get_points_for_x(point)
                            points = self.coordinate_translator.marker_to_camera(
                                points, rvec, tvec
                            )
                            points = self.coordinate_translator.camera_to_img(points)
                            img = self._draw_x(img, points, color="white")

                    # Draw points in global coordinates
                    for point in self.display_points_in_global_coord:
                        points = self._get_points_for_x(point)
                        points = self.coordinate_translator.global_to_marker(
                            points, marker_id
                        )
                        points = self.coordinate_translator.marker_to_camera(
                            points, rvec, tvec
                        )
                        points = self.coordinate_translator.camera_to_img(points)
                        img = self._draw_x(img, points, color="yellow")

                # Draw averaged global points
                for point in self.display_points_in_global_coord:
                    points = self._get_points_for_x(point, size=0.02)
                    points = self._global_points_from_detected_markers(
                        points, ids, rvecs, tvecs
                    )
                    points = self.coordinate_translator.camera_to_img(points)
                    img = self._draw_x(img, points, color="cyan", thickness=2)
        return img

    def _global_points_from_detected_markers(self, points, ids, rvecs, tvecs):
        camera_points = []

        for marker_id, rvec, tvec in zip(ids, rvecs, tvecs):
            points_translated = self.coordinate_translator.global_to_marker(
                points, marker_id[0]
            )
            points_translated = self.coordinate_translator.marker_to_camera(
                points_translated, rvec, tvec
            )
            camera_points.append(points_translated)

        camera_points = np.median(np.array(camera_points), axis=0)
        return camera_points

    def _get_points_for_x(self, center, size=0.01):
        arr = np.array([center] * 6, dtype=np.float32)
        arr[0, 0] += size
        arr[1, 0] -= size
        arr[2, 1] += size
        arr[3, 1] -= size
        arr[4, 2] += size
        arr[5, 2] -= size

        return arr

    def _draw_x(self, img, x_points, color="white", thickness=1):
        col = color_to_rgb[color]
        img = cv2.line(img, x_points[0], x_points[1], col, thickness)
        img = cv2.line(img, x_points[2], x_points[3], col, thickness)
        img = cv2.line(img, x_points[4], x_points[5], col, thickness)
        return img

    def _state_to_display(self, state):
        img = np.zeros((800, 300, 3), np.uint8)

        img = draw_text(img, "velocity:", (10, 30))
        for i, n in enumerate(self.v_names):
            img = draw_text(img, f"{n}={state[n]}", (10, 30 * (2 + i)))

        img = draw_text(img, "acceleration:", (10, 180))
        for i, n in enumerate(self.a_names):
            img = draw_text(img, f"{n}={state[n]}", (10, 30 * (7 + i)))

        img = draw_text(img, "angles:", (10, 330))
        for i, n in enumerate(self.rpy_names):
            img = draw_text(img, f"{n}={state[n]}", (10, 30 * (12 + i)))

        img = draw_text(img, "height:", (10, 480))
        for i, n in enumerate(self.h_names):
            img = draw_text(img, f"{n}={state[n]}", (10, 30 * (17 + i)))

        img = draw_text(img, "other:", (10, 630))
        for i, n in enumerate(self.other_names):
            img = draw_text(img, f"{n}={state[n]}", (10, 30 * (22 + i)))

        return img

    def __del__(self):
        cprint("\rShutting down tello, please wait (1/3) ...", "yellow", end="")
        self.tello.end()
        cprint("\rShutting down tello, please wait (2/3) ...", "yellow", end="")
        cv2.destroyAllWindows()
        cprint("\rFinished shutting down tello (3/3)" + " " * 20, "yellow", end="\n")
