from lib.tello_morelo import Tello


def main():
    tello = Tello(
        path_to_calibration_images="./images_for_calibration",
        display_points_dict={
            0: [
                (0.1, 0.1, 0.1),
                (0.05, 0.05, 0.05),
            ]
        },
    )

    while True:
        tello.get_data(interactive=True)


if __name__ == "__main__":
    main()
