from tello_morelo import Tello
from utils import connect_or_exit
import os
import cv2


path_to_folder_for_images = "images_for_calibration"
num_images_for_calibration = 30
chessboard_shape = (8, 5)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

def main():
    os.makedirs(path_to_folder_for_images, exist_ok=True) 
    images_saved = 0 
    connect_or_exit() 
    tello = Tello()
    
    while images_saved < num_images_for_calibration:
        img, _ = tello.get_data(display=False)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, chessboard_shape)
        if ret:
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            img_to_save = img.copy()
            cv2.drawChessboardCorners(img, chessboard_shape, corners2, ret)
            cv2.imshow("image to save", img)
            print("Press 's' to save or anything else to skip")
            
            if ord("s") == cv2.waitKey() & 0xFF:
                path = os.path.join(path_to_folder_for_images, f"{images_saved}.jpg")
                cv2.imwrite(path, img_to_save)
                images_saved += 1
                print("images saved", images_saved)
        else:
            cv2.imshow("image to save", img)
            cv2.pollKey()


if __name__ == "__main__":
    main()
