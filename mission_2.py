from lib.marker_maps import room_map
from lib.tello_morelo import Tello


def main():
    tello = Tello(
        path_to_calibration_images="./images_for_calibration",
        display_points_in_marker_coord={},
        display_points_in_global_coord=[
            (1, 0.2, 0.5),
        ],
        marker_id_to_coordinate_change=room_map,
    )

    while True:
        tello.get_data(interactive=True)


if __name__ == "__main__":
    main()
