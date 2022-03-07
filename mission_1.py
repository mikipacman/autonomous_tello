from lib.tello_morelo import Tello

# MISSION 1
# The goal is to fly through a cardboard door that has aruco markers on

# TODO when stats in manouvers will be implemented check how a accurate they are
# adn whether the center of tello's coordinate system is in camera's coordinate system

# TODO check whether moving with fly_to_point_in_camera_coord() is reliable
# TODO handle errors like respone "error Not joystick"


def main():
    x_center = 0.12 + 0.38 / 2
    points = [
        (x_center, 0, 0),
        (x_center, 0, 0.5),
        (x_center, 0, -0.5),
    ]

    tello = Tello(
        path_to_calibration_images="./images_for_calibration",
        display_points_dict={0: points},
    )

    tello.takeoff()
    found_marker = tello.find_any_markers([0], rotation="clockwise")
    tello.fly_to_point_in_marker_coord((x_center, -0.2, 0.5), 0, velocity=40)
    tello.fly_to_point_in_camera_coord((0, 0, 1))
    tello.land()


if __name__ == "__main__":
    main()
