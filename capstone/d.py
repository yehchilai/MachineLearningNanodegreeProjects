import os
from urllib.request import urlopen
# from io import StringIO # Python2
from io import BytesIO # Python3
from PIL import Image
from tqdm import tqdm
from multiprocessing import Pool
import multiprocessing as mp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

train_data = pd.read_csv('data/train/train.csv')
# print(train_data.shape)
test_data = pd.read_csv('data/test/test.csv')
# print(test_data.shape)

def downloadImage(row):
#     print(row)
    image_id = row[0]
    url = row[1] 
    landmark_id = row[2] 
    image_folder = row[3]
    outfile = row[4]
    file = '{x}.{y}'.format(x=image_id,y='jpg')
    filename = os.path.join(image_folder, file)
    
    if os.path.exists(filename):
        print('Image %s already exists. Skipping download.' % filename)
        return
    try:
        response = urlopen(url)
        image_data = response.read()
    except:
        print('Warning: Could not download image %s from %s' % (image_id, url))
        return
    
    try:
        pil_image = Image.open(BytesIO(image_data))
#         pil_image.thumbnail(size)
    except:
        print('Warning: Failed to parse image %s' % image_id)
        return

    try:
        pil_image_resize = pil_image.resize(size=(224,224))
    except:
        print('Warning: Failed to resize image %s' % image_id)
        return

    try:
        pil_image_rgb = pil_image_resize.convert('RGB')
    except:
        print('Warning: Failed to convert image %s to RGB' % image_id)
        return

    try:
        pil_image_rgb.save(filename, format='JPEG', quality=90)
        with open(outfile, 'a') as f:
            f.write('{x} {y} {z}\n'.format(x=image_id, y=landmark_id, z=filename))
    except:
        print('Warning: Failed to save image %s' % filename)
        return

def downloadTrainData(dataframe, image_folder, outfile):
    if not os.path.exists(image_folder):
        os.mkdir(image_folder)
    rowlist = []
#     for i in range(len(dataframe['id'])):
    for i in range(20):
        t = (dataframe.iloc[i].id, dataframe.iloc[i].url, dataframe.iloc[i].landmark_id, image_folder, outfile)
        rowlist.append(t)
    print('## complete processing rowlist.')
    with Pool(processes=mp.cpu_count() - 1) as p:
        print('open pool')
        r = list(tqdm(p.imap(downloadImage, rowlist), total=len(rowlist)))
        
def downloadTestData(dataframe, image_folder, outfile):
    if not os.path.exists(image_folder):
        os.mkdir(image_folder)
    rowlist = []
#     for i in range(len(dataframe['id'])):
    for i in range(20):
        t = (dataframe.iloc[i].id, dataframe.iloc[i].url, 'None', image_folder, outfile)
        rowlist.append(t)
    print('## complete processing rowlist.')
    with Pool(processes=mp.cpu_count() - 1) as p:
        print('open pool')
        r = list(tqdm(p.imap(downloadImage, rowlist), total=len(rowlist)))
                 
if __name__ == '__main__':
    downloadTrainData(train_data, 'data/trainImages/', 'data/trainImagesList')
    downloadTestData(test_data, 'data/testImages/', 'data/testImagesList')
#     downloadData(train_data, 'data/t/', 'data/tList')