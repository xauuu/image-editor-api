import random
import string
import numpy as np
import urllib.request
import cv2
import base64

letters = string.ascii_lowercase


def str_id():
    return ''.join(random.choice(letters) for i in range(10))


def url_to_image(url):
    url_response = urllib.request.urlopen(url)
    img_array = np.array(bytearray(url_response.read()), dtype=np.uint8)
    img = cv2.imdecode(img_array, -1)
    return img

def readb64(uri):
   encoded_data = uri.split(',')[1]
   nparr = np.fromstring(base64.b64decode(encoded_data), np.uint8)
   img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
   return img

def export_image(img, path):
    cv2.imwrite(path, img)
