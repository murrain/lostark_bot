from ctypes.wintypes import MAX_PATH
import pyautogui as pg
import cv2
import time
import glob
import os
import numpy as np
import pytesseract
from matplotlib import pyplot as plt
from PIL import ImageGrab, Image

pytesseract.pytesseract.tesseract_cmd = r'C:\Users\patrick\AppData\Local\Programs\Tesseract-OCR\tesseract'

MINIMAP_REGION = (2234,219,2525,475)
HP_REGION = (1008,1137,1061,1155)
TEMPLATES = {
    "boss":     {"template": cv2.cvtColor(cv2.imread("boss.png"),cv2.COLOR_RGB2GRAY), "color": (200,0,0)},
    "miniboss": {"template": cv2.cvtColor(cv2.imread("miniboss.png"),cv2.COLOR_RGB2GRAY), "color": (0,100,200)}, 
    "enemy":    {"template": cv2.cvtColor(cv2.imread("enemy.png"),cv2.COLOR_RGB2GRAY), "color": (99,0,99)}
}

MAXHP = 77156

def capture_minimap():
    return ImageGrab.grab(bbox=MINIMAP_REGION)

def capture_hp():
    return ImageGrab.grab(bbox=HP_REGION)

def write_image(filename,image):
    image = cv2.cvtColor(np.array(image),cv2.COLOR_RGB2BGR)
    cv2.imwrite(filename, image)

def find_boss_miniboss_enemy(image):
    drawn_image = image
    image = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    for key,template in TEMPLATES.items():
        print("Looking for " + key)
        res = cv2.matchTemplate(image,template["template"],cv2.TM_CCOEFF_NORMED)
        threshold = 0.54
        h,w = template["template"].shape
        location = np.where( res >= threshold)
        nms_location = non_max_suppression_fast(np.array(list(zip(*location[::-1]))),h,w,0.3)
        for pt in nms_location:
            cv2.rectangle(drawn_image,pt,(pt[0] + w, pt[1] + h),template["color"],2)
            print("{} found at location {}".format(key,pt))
    return drawn_image

# Malisiewicz et al.
def non_max_suppression_fast(boxes, h,w,overlapThresh):
	# if there are no boxes, return an empty list
	if len(boxes) == 0:
		return []
	# if the bounding boxes integers, convert them to floats --
	# this is important since we'll be doing a bunch of divisions
	if boxes.dtype.kind == "i":
		boxes = boxes.astype("float")
	# initialize the list of picked indexes	
	pick = []
	# grab the coordinates of the bounding boxes
	x1 = boxes[:,0]
	y1 = boxes[:,1]
	x2 = boxes[:,0]+w
	y2 = boxes[:,1]+h
	# compute the area of the bounding boxes and sort the bounding
	# boxes by the bottom-right y-coordinate of the bounding box
	area = (x2 - x1 + 1) * (y2 - y1 + 1)
	idxs = np.argsort(y2)
	# keep looping while some indexes still remain in the indexes
	# list
	while len(idxs) > 0:
		# grab the last index in the indexes list and add the
		# index value to the list of picked indexes
		last = len(idxs) - 1
		i = idxs[last]
		pick.append(i)
		# find the largest (x, y) coordinates for the start of
		# the bounding box and the smallest (x, y) coordinates
		# for the end of the bounding box
		xx1 = np.maximum(x1[i], x1[idxs[:last]])
		yy1 = np.maximum(y1[i], y1[idxs[:last]])
		xx2 = np.minimum(x2[i], x2[idxs[:last]])
		yy2 = np.minimum(y2[i], y2[idxs[:last]])
		# compute the width and height of the bounding box
		w = np.maximum(0, xx2 - xx1 + 1)
		h = np.maximum(0, yy2 - yy1 + 1)
		# compute the ratio of overlap
		overlap = (w * h) / area[idxs[:last]]
		# delete all indexes from the index list that have
		idxs = np.delete(idxs, np.concatenate(([last],
			np.where(overlap > overlapThresh)[0])))
	# return only the bounding boxes that were picked using the
	# integer data type
	return boxes[pick].astype("int")

if __name__ == "__main__":
    images = glob.glob('unmarked\*.png')
    for imagepath in images:
        imagename = os.path.basename(imagepath)
        print("Searching {}".format(imagename))
        image = cv2.imread(imagepath)

        h,w = image.shape[:2]
        mask = np.zeros((h,w),np.uint8)

        rect = ( int(h/2 - 12), int(w/2 - 12),int(h/2 + 12), int(w/2 + 12) )
        bgModel = np.zeros((1,65),np.float64);
        fgModel = np.zeros((1,65),np.float64);

        #cv2.grabCut(image,mask,rect,bgModel,fgModel,5)
        #mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
        #masked_image = image*mask2[:,:,np.newaxis]

        #player_location = tuple(map(lambda i, j: (i + j) / 2, player_min_loc, player_max_loc))
        player_location = (w/2, h/2)  
        #print("Player found at: {}".format(player_location))
        image = find_boss_miniboss_enemy(image)
        cv2.imwrite("marked\marked_" + imagename, image)
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        plt.imshow(image),plt.colorbar(),plt.show()	