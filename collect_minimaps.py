import pyautogui as pg
import os
import cv2
import time
import numpy as np
from PIL import ImageGrab, Image

MINIMAP_REGION = (2265,216,270,227)
HP_REGION = ()

def capture_minimap():
    return pg.screenshot(region=MINIMAP_REGION)

def capture_hp():
    return pg.screenshot(region=HP_REGION)

if __name__ == "__main__":
    img_count = 0
    previous_hash = 0

    while img_count < 40:

        minimap_filename = str(int(time.time())) + ".png"
        minimap_image = capture_minimap()
        minimap_image = cv2.cvtColor(np.array(minimap_image),
                     cv2.COLOR_RGB2BGR)
        cv2.imwrite(minimap_filename, minimap_image)
        img_count +=1

        time.sleep(3)