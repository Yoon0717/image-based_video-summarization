import json
import os
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2

path = "C:/Users/sunp/Desktop/nlp/TS_105_Larva_S01_016/"

csv = pd.read_csv("C:/Users/sunp/Desktop/nlp/lava_01_16.csv")
texts = list(csv['text'])

for i in range(len(csv['file_name'])):
    cv2.imshow(csv['file_name'][i], cv2.imread(path + csv['file_name'][i], cv2.IMREAD_COLOR))
    cv2.waitKey(0)
    texts[i] = input()
    cv2.destroyAllWindows()

data = {
    "file_name" : csv['file_name'],
    "text" : texts
}

df = pd.DataFrame(data)

df.to_csv("C:/Users/sunp/Desktop/nlp/laval.csv")
