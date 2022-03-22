# Tello Morelo
This repo implements interface for autonomous control of Tello drone using basic computer vision algorithms.
The interface is high level and allows for designing 'missions' for the drone (see example mission code in `mission_1.py`).
The interface plots what the drone 'sees' in real-time. That allows for quick debugging and nice visualizations (see a gif below).

So far there is only one mission implemented, but as I add more features there will be more ;)

## Mission 1
Fly through a cardboard gate that has an aruco marker on. 

Here is a visualization of what the drone sees and how it looks from third perspective.


![alt text](./gifs/tello_flying_through_gate.gif)

# Future Ideas
- SLAM
- Monocular depth estimation
- Pose estimation
- Object detection
