import numpy as np
from scipy.spatial.transform import Rotation

from lib.image_processing import CoordinateChange


class MarkerMap:
    def __init__(self, id_to_marker_map):
        self.id_to_marker_map = id_to_marker_map

    @staticmethod
    def init_marker_plane(
        n_rows, n_cols, x_delta, y_delta, start_id, id_x_delta, id_y_delta
    ):
        """Init marker plane with given parameters.

        For example (n_rows, n_cols, x_delta, y_delta, start_id, id_x_delta, id_y_delta) = (3, 2, 0.2, 0.1, 0, 1, 2)
        will create object that this describes this marker plane:

        4 5
        2 3
        0 1

        where horizontal and vertical delta are 0.2 and 0.1 respectively.
        """
        id_to_marker_map = {}

        for i in range(n_rows):
            for j in range(n_cols):
                id = start_id + i * id_y_delta + j * id_x_delta
                id_to_marker_map[id] = CoordinateChange(
                    rotation=Rotation.identity(),
                    translation=np.array([x_delta * j, y_delta * i, 0]),
                )

        return MarkerMap(id_to_marker_map)

    def translate(self, translation):
        id_to_marker_map = {}
        for id, marker in self.id_to_marker_map.items():
            id_to_marker_map[id] = CoordinateChange(
                rotation=marker.rotation,
                translation=marker.translation + np.array(translation),
            )

        return MarkerMap(id_to_marker_map)

    def rotate(self, rotation):
        id_to_marker_map = {}
        for id, marker in self.id_to_marker_map.items():
            id_to_marker_map[id] = CoordinateChange(
                rotation=marker.rotation * rotation,
                translation=rotation.as_matrix() @ marker.translation,
            )

        return MarkerMap(id_to_marker_map)

    def __add__(self, other):
        id_to_marker_map = {}
        for id, marker in self.id_to_marker_map.items():
            id_to_marker_map[id] = marker

        for id, marker in other.id_to_marker_map.items():
            id_to_marker_map[id] = marker

        return MarkerMap(id_to_marker_map)


# Room map. marker_side = 0.059
room_map_plane_1 = MarkerMap.init_marker_plane(
    n_rows=5,
    n_cols=4,
    x_delta=0.35,
    y_delta=0.25,
    start_id=12,
    id_x_delta=1,
    id_y_delta=6,
).translate([0.18, 0.5, 0])

room_map_plane_2 = (
    MarkerMap.init_marker_plane(
        n_rows=5,
        n_cols=2,
        x_delta=0.33,
        y_delta=0.25,
        start_id=10,
        id_x_delta=1,
        id_y_delta=6,
    )
    .rotate(Rotation.from_euler("y", 90, degrees=True))
    .translate([0, 0.5, 0.46])
)

room_map = room_map_plane_1 + room_map_plane_2
