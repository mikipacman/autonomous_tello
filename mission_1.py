from lib.tello_morelo import Tello

# MISSION 1
# The goal is to fly through a cardboard door that has aruco markers on

# TODO


def main():
    points = [
        (0.12 + 0.38 / 2, 0, 0),
        (0.12 + 0.38 / 2, 0, 0.5),
        (0.12 + 0.38 / 2, 0, -0.5),
    ]

    tello = Tello(
        path_to_calibration_images="./images_for_calibration",
        display_points_dict={0: points},
    )

    tello.takeoff()
    found_marker = tello.find_any_markers([0], rotation="clockwise")
    points[1][1] -= 0.1
    tello.fly_to_point_in_marker_coord(points[1], 0, velocity=20)
    tello.land()


if __name__ == "__main__":
    main()
