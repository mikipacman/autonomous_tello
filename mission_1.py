from termcolor import cprint

from lib.tello_morelo import Tello

# MISSION 1
# The goal is to fly through a cardboard door that has aruco markers on


def main():
    tello = Tello(
        path_to_calibration_images="./images_for_calibration",
        display_points_dict={
            0: [
                (0.1, 0.1, 0.1),
                (0.05, 0.05, 0.05),
                (0, 0, 0.5),
            ]
        },
    )

    # TODO
    # Center to marker

    # usefull api funcs:
    # set_speed()
    # go_xyz_speed(self, x, y, z, speed)
    # curve_xyz_speed(self, x1, y1, z1, x2, y2, z2, speed)
    # rotate_clockwise(self, x)
    # rotate_counter_clockwise(self, x)
    # move(self, direction, x)

    tello.takeoff()
    found_marker = tello.find_any_markers([0], rotation="clockwise")
    tello.sleep(1)
    tello.fly_to_point_in_marker_coord((0, 0, 0.3), 0)
    tello.sleep(5)
    tello.land()


if __name__ == "__main__":
    main()
