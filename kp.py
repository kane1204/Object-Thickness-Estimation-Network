from pickle import NONE
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import mplcursors
from os import listdir
from os.path import isfile, join
import pandas as pd
dir = "data\human_samples"
imgs = []
for cat in listdir(dir):
    for model in listdir(join(dir, cat)):
        for img in listdir(join(dir, cat, model)):
            imgs.append(join(cat, model, img))

print(imgs)
d = {"file_name": imgs, "thicker":None, "thinner":None}
real_df = pd.DataFrame(d)
print(real_df)

idx = 0
for imgpath in imgs:
    image = Image.open(join(dir, imgpath,"img.png"))
    thick = np.load(join(dir, imgpath,"thicc_map.npy"))
    new_im = Image.fromarray(thick)
    image.paste(new_im, (128,0))
    data = np.array(new_im)
    img = plt.imshow(data)
    plt.title(imgpath)
    points = []

    cursor = mplcursors.cursor(img, hover=False)
    @cursor.connect("add")
    def cursor_clicked(sel):       
        # sel.annotation.set_visible(False)
        sel.annotation.set_text(
            f'Clicked on\nx: {sel.target[0]:.2f} y: {sel.target[1]:.2f}')
        points.append(sel.index)
       

    plt.show()
    print(f"Thick Point:", points[0], "Thin Point:", points[1])
    real_df['thicker'][idx]= points[0]
    real_df['thinner'][idx]= points[1]
    idx+=1


print(real_df)
real_df.to_csv("realdata.csv")