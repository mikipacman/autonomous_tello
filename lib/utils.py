import collections
import time
from subprocess import DEVNULL, check_call

import cv2
from termcolor import cprint


# Wifi
def connect_to_wifi(name, password, num_retries=20, interface="wlp4s0"):
    cmd_refresh_list = f"sudo iwlist {interface} scan"
    cmd_connect = "nmcli d wifi connect {} password {}".format(name, password)

    for i in range(num_retries):
        try:
            check_call(cmd_refresh_list.split(), stdout=DEVNULL, stderr=DEVNULL)
            check_call(cmd_connect.split(), stdout=DEVNULL, stderr=DEVNULL)
        except:
            cprint(
                f"\rCould not connect to {name}. Retrying {i + 1}/{num_retries} ...",
                "yellow",
                end="",
            )
            continue
        if i:
            print()
        return True

    print()
    return False


def connect_or_exit(server_name, password):
    if not connect_to_wifi(server_name, password):
        cprint(f"Failed to connect to wifi {server_name}", "red")
        exit(1)
    else:
        cprint(f"Connected to {server_name}", "green")


# Fps
class FPS:
    def __init__(self, avarageof=200):
        self.frametimestamps = collections.deque(maxlen=avarageof)

    def __call__(self):
        self.frametimestamps.append(time.time())
        if len(self.frametimestamps) > 1:
            return len(self.frametimestamps) / (
                self.frametimestamps[-1] - self.frametimestamps[0]
            )
        else:
            return 0.0


# cv2 drawing utils
color_to_rgb = {
    "white": (255, 255, 255),
    "red": (0, 0, 255),
    "blue": (255, 0, 0),
    "black": (0, 0, 0),
    "green": (0, 255, 0),
    "yellow": (0, 255, 255),
    "pink": (255, 0, 255),
    "cyan": (255, 255, 0),
}


def draw_text(img, text, pos, color="white"):
    return cv2.putText(
        img,
        text,
        pos,
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        color_to_rgb[color],
        2,
        cv2.LINE_AA,
    )


# Decorators
def assert_input_shapes(*shapes):
    def dec(f):
        def f_to_ret(*args):
            assert len(args) == len(
                shapes
            ), f"Assert on shapes in {f.__name__} cannot be matched"
            for i, (a, s) in enumerate(zip(args, shapes)):
                if not s:
                    continue
                assert (
                    a.shape == s
                ), f"Shapes do not match in {f.__name__} for {i} argument"

            return f(*args)

        return f_to_ret

    return dec


def assert_output_shapes(shape):
    def dec(f):
        def f_to_ret(*args):
            ret = f(*args)
            assert (
                ret.shape == shape
            ), f"Shape of output do not match in {f.__name__}. The shape: {ret.shape}"
            return ret

        return f_to_ret

    return dec
