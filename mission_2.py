from scipy.spatial.transform import Rotation

from lib.image_processing import CoordinateChange
from lib.tello_morelo import Tello


def main():
    eps = 0.0875
    tello = Tello(
        path_to_calibration_images="./images_for_calibration",
        display_points_in_marker_coord={4: [(0, 0, 0.1)]},
        display_points_in_global_coord=[(0.2, 0, 0.1)],
        marker_id_to_coordinate_change={
            4: CoordinateChange(
                rotation=Rotation.from_euler("x", 90, degrees=True),
                translation=[0, 0.055, 0.135],
            ),
            6: CoordinateChange(
                rotation=Rotation.identity(),
                translation=[0, 0, 0],
            ),
            7: CoordinateChange(
                rotation=Rotation.identity(),
                translation=[eps, 0, 0],
            ),
            8: CoordinateChange(
                rotation=Rotation.identity(),
                translation=[2 * eps, 0, 0],
            ),
            9: CoordinateChange(
                rotation=Rotation.identity(),
                translation=[0, -eps, 0],
            ),
        },
    )
    tello.sleep(int(1e10))


if __name__ == "__main__":
    main()
