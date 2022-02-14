import enum
import logging
import cv2
import djitellopy
from numpy.lib.function_base import select
import matplotlib.pyplot as plt
import collections
import numpy as np


# Set logging level
djitellopy.Tello.LOGGER.setLevel(logging.WARNING)


class Tello:

    def __init__(self):
        self.tello = djitellopy.Tello()
        self.cv2_camera_window_name = "tello"
        self.cv2_state_window_name = "state" 
        
        # Things for plotting data
        self.data_queue_capacity = 300 
        self.data_queue = collections.deque(maxlen=self.data_queue_capacity)
        self.state_names = [
            'pitch', 'roll', 'yaw',
            'vgx', 'vgy', 'vgz',
            'templ', 'temph', 'tof', 'h', 'bat', 'baro', 'time',
            'agx', 'agy', 'agz'
        ]
        self.fig, self.ax = plt.subplots(nrows=len(self.state_names), figsize=(5, 10))
        self.lines = [ax.plot(
            np.arange(self.data_queue_capacity),
            np.zeros(self.data_queue_capacity),
        )[0] for ax in self.ax]
        for i, name in enumerate(self.state_names):
            self.ax[i].set_xlabel(name)
        self.frame_count = 0 
        plt.tight_layout(pad=0)
        self.tello.connect()
      
        # TODO this does not work, can we fix it?
        # self.tello.set_video_direction(djitellopy.Tello.CAMERA_FORWARD)
        # self.tello.set_video_bitrate(djitellopy.Tello.BITRATE_1MBPS)
        # self.tello.set_video_fps(djitellopy.Tello.FPS_5)

        self.tello.streamoff()
        self.tello.streamon()


    def get_data(self):
        frame_read = self.tello.get_frame_read()
        state = self.tello.get_current_state()
        self.data_queue.append((frame_read, state)) 
        self.frame_count += 1
        
        cv2.imshow(self.cv2_camera_window_name, frame_read.frame)
        if self.frame_count % 100 == 0:
            cv2.imshow(self.cv2_state_window_name, self._get_state_plot()) 
        
        cv2.pollKey()
        
        return frame_read.frame, state

    def _get_state_plot(self):
        for i, k in enumerate(self.state_names):  
            y = [state[k] for (_, state) in self.data_queue] 
            y += [0] * (self.data_queue_capacity - len(y)) 
            self.lines[i].set_ydata(list(y))
            self.ax[i].set_ylim([min(y) - 1, max(y) + 1]) 
        
        self.fig.canvas.draw()
       
        img = np.fromstring(self.fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        img = img.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img
