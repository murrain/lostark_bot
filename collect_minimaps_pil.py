from ctypes.wintypes import MAX_PATH
import pyautogui as pg
import os
import cv2
import time
import numpy as np
import pytesseract
from PIL import ImageGrab, Image

pytesseract.pytesseract.tesseract_cmd = r'C:\Users\patrick\AppData\Local\Programs\Tesseract-OCR\tesseract'

MINIMAP_REGION = (2234,219,2525,475)
HP_REGION = (1008,1137,1061,1155)
TEMPLATES = {"boss": "boss.png","miniboss": "miniboss.png", "enemy": "enemy.png"}

MAXHP = 77156

def capture_minimap():
    return ImageGrab.grab(bbox=MINIMAP_REGION)

def capture_hp():
    return ImageGrab.grab(bbox=HP_REGION)

def write_image(filename,image):
    image = cv2.cvtColor(np.array(image),cv2.COLOR_RGB2BGR)
    cv2.imwrite(filename, image)
    

if __name__ == "__main__":
    img_count = 0
    previous_hash = 0

    while img_count < 40:

        now = str(int(time.time()))

        minimap_filename = "unmarked\minimap_" + now +".png"
        minimap_image = capture_minimap()
        write_image(minimap_filename,minimap_image)
        hp_image = capture_hp()
        #write_image("hp_" + now + ".png",hp_image)
        hp = pytesseract.image_to_string(cv2.cvtColor(np.array(hp_image),cv2.COLOR_RGB2BGR))
        try:
           hp = float(hp)
           print("{} / {} ({}%)".format (hp,MAXHP, hp / MAXHP * 100))
        except:
            print("UNABLE TO READ HP")
        img_count +=1

        time.sleep(2)