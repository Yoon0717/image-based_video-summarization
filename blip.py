from datasets import load_dataset
import pandas as pd
import os
import numpy as np
from PIL import Image

image_path = 'C:/Users/sunp/Desktop/nlp/TS_105_Larva_S01_016/'
textdata = pd.read_csv("C:/Users/sunp/Desktop/nlp/lava_01_16.csv")

img_list = os.listdir(image_path) #디렉토리 내 모든 파일 불러오기
img_list_jpg = [img for img in img_list if img.endswith(".jpg")] #지정된 확장자만 필터링
img_list_jpg = sorted(img_list_jpg)
modified_filenames = [filename.replace('.jpg', '') for filename in img_list_jpg if filename.replace('.jpg', '')]
img_list_jpg = [name for name in img_list_jpg if name.replace('.jpg', '') in modified_filenames]

t_data = []
textdata = pd.read_csv('C:/Users/sunp/Desktop/nlp/lava_01_16.csv')
fnames = []

for i in range(len(textdata)):
  if textdata['file_name'][i] in img_list_jpg:
    fnames.append(textdata['file_name'][i])
    t_data.append(textdata['text'][i])

print(len(fnames))
print(len(t_data))

from transformers import BlipProcessor, BlipForConditionalGeneration

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")

from torch.utils.data import Dataset, DataLoader
from PIL import Image

class ImageCaptioningDataset(Dataset):
    def __init__(self, images, texts, processor):
        self.images = images
        self.texts = texts
        self.processor = processor

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(image_path + self.images[idx])
        text = self.texts[idx]
        encoding = self.processor(images=image, text=text, padding="max_length", return_tensors="pt")
        # remove batch dimension
        encoding = {k:v.squeeze() for k,v in encoding.items()}
        return encoding

train_dataset = ImageCaptioningDataset(fnames,t_data,processor)
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=4)

import torch

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

model.train()

for epoch in range(2):
  print("Epoch:", epoch)
  for idx, batch in enumerate(train_dataloader):
    input_ids = batch.pop("input_ids").to(device)
    pixel_values = batch.pop("pixel_values").to(device)

    outputs = model(input_ids=input_ids,
                    pixel_values=pixel_values,
                    labels=input_ids)

    loss = outputs.loss

    print("Loss:", loss.item())

    loss.backward()

    optimizer.step()
    optimizer.zero_grad()

captions = []


from PIL import Image

for i in range(len(fnames)):
    image = Image.open(image_path+fnames[i])
    inputs = processor(images=image, return_tensors="pt").to(device)
    pixel_values = inputs.pixel_values
    generated_ids = model.generate(pixel_values=pixel_values, max_length=50)
    generated_caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    captions.append(generated_caption)

import pandas as pd

data = {
    "captions" : captions
}

df = pd.DataFrame(data)

df.to_csv("C:/Users/sunp/Desktop/nlp/cap.csv")