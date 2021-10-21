# -*- coding: utf-8 -*-
"""
Created on Tue May 25 14:28:40 2021

@author: sebas
"""

import numpy as np
import cv2
import pandas as pd
from shapely.wkt import loads as wkt_loads
import tifffile as tiff
import os
import random
from sklearn.metrics import jaccard_score


import zipfile

os.chdir("C:/Users/sebas/Documents/2021-10/Vision/VC_Project/")
N_Cls = 1
inDir = 'C:/Users/sebas/Documents/2021-10/Vision/VC_Project/'
#inDir = '/home/n01z3/dataset/dstl'
DF = pd.read_csv(inDir + '/train_wkt_v4.csv.zip')
GS = pd.read_csv(inDir + '/grid_sizes.csv.zip', names=['ImageId', 'Xmax', 'Ymin'], skiprows=1)
SB = pd.read_csv(os.path.join(inDir, 'sample_submission.csv.zip'))
ISZ = 160
smooth = 1e-12

#os.mkdir('C:/Users/sebas/Documents/2021-10/Vision/VC_Project/kaggle/data/')
#os.mkdir('C:/Users/sebas/Documents/2021-10/Vision/VC_Project/kaggle/msk/')
#os.mkdir('C:/Users/sebas/Documents/2021-10/Vision/VC_Project/kaggle/weights/')
#os.mkdir('C:/Users/sebas/Documents/2021-10/Vision/VC_Project/kaggle/subm/')

def _convert_coordinates_to_raster(coords, img_size, xymax):
    # __author__ = visoft
    # https://www.kaggle.com/visoft/dstl-satellite-imagery-feature-detection/export-pixel-wise-mask
    Xmax, Ymax = xymax
    H, W = img_size
    W1 = 1.0 * W * W / (W + 1)
    H1 = 1.0 * H *  H / (H + 1)
    xf = W1 / Xmax
    yf = H1 / Ymax
    coords[:, 1] *= yf
    coords[:, 0] *= xf
    coords_int = np.round(coords).astype(np.int32)
    return coords_int


def _get_xmax_ymin(grid_sizes_panda, imageId):
    # __author__ = visoft
    # https://www.kaggle.com/visoft/dstl-satellite-imagery-feature-detection/export-pixel-wise-mask
    xmax, ymin = grid_sizes_panda[grid_sizes_panda.ImageId == imageId].iloc[0, 1:].astype(float)
    return (xmax, ymin)


def _get_polygon_list(wkt_list_pandas, imageId, cType):
    # __author__ = visoft
    # https://www.kaggle.com/visoft/dstl-satellite-imagery-feature-detection/export-pixel-wise-mask
    df_image = wkt_list_pandas[wkt_list_pandas.ImageId == imageId]
    multipoly_def = df_image[df_image.ClassType == cType].MultipolygonWKT
    polygonList = None
    if len(multipoly_def) > 0:
        assert len(multipoly_def) == 1
        polygonList = wkt_loads(multipoly_def.values[0])
    return polygonList


def _get_and_convert_contours(polygonList, raster_img_size, xymax):
    # __author__ = visoft
    # https://www.kaggle.com/visoft/dstl-satellite-imagery-feature-detection/export-pixel-wise-mask
    perim_list = []
    interior_list = []
    if polygonList is None:
        return None
    for k in range(len(polygonList)):
        poly = polygonList[k]
        perim = np.array(list(poly.exterior.coords))
        perim_c = _convert_coordinates_to_raster(perim, raster_img_size, xymax)
        perim_list.append(perim_c)
        for pi in poly.interiors:
            interior = np.array(list(pi.coords))
            interior_c = _convert_coordinates_to_raster(interior, raster_img_size, xymax)
            interior_list.append(interior_c)
    return perim_list, interior_list


def _plot_mask_from_contours(raster_img_size, contours, class_value=1):
    # __author__ = visoft
    # https://www.kaggle.com/visoft/dstl-satellite-imagery-feature-detection/export-pixel-wise-mask
    img_mask = np.zeros(raster_img_size, np.uint8)
    if contours is None:
        return img_mask
    perim_list, interior_list = contours
    cv2.fillPoly(img_mask, perim_list, class_value)
    cv2.fillPoly(img_mask, interior_list, 0)
    return img_mask


def generate_mask_for_image_and_class(raster_size, imageId, class_type, grid_sizes_panda=GS, wkt_list_pandas=DF):
    # __author__ = visoft
    # https://www.kaggle.com/visoft/dstl-satellite-imagery-feature-detection/export-pixel-wise-mask
    xymax = _get_xmax_ymin(grid_sizes_panda, imageId)
    polygon_list = _get_polygon_list(wkt_list_pandas, imageId, class_type)
    contours = _get_and_convert_contours(polygon_list, raster_size, xymax)
    mask = _plot_mask_from_contours(raster_size, contours, 1)
    return mask


def M(image_id):
    # __author__ = amaia
    # https://www.kaggle.com/aamaia/dstl-satellite-imagery-feature-detection/rgb-using-m-bands-example
    zip_path = 'C:/Users/sebas/Documents/2021-10/Vision/VC_Project/sixteen_band.zip'
    tgtImg = '{}_M.tif'.format(image_id)
    with zipfile.ZipFile(zip_path) as myzip:
        files_in_zip = myzip.namelist()
        for fname in files_in_zip:
            if fname.endswith(tgtImg):
                with myzip.open(fname) as myfile:
                    img = tiff.imread(myfile)
                    img = np.rollaxis(img, 0, 3)
                    return img


#def normalize(bands, lower_percent=5, higher_percent=95):
#    out = np.zeros_like(bands)
#    n = bands.shape[2]
#    for i in range(n):
#        a = 0  # np.min(band)
#        b = 1  # np.max(band)
#        c = np.percentile(bands[:, :, i], lower_percent)
#        d = np.percentile(bands[:, :, i], higher_percent)
#        t = a + (bands[:, :, i] - c) * (b - a) / (d - c)
#        t[t < a] = a
#        t[t > b] = b
#        out[:, :, i] = t

#    return out.astype(np.uint8)


def normalize(img):
    min = img.min()
    max = img.max()
    x = (img - min) / (max - min)
    return x

def stick_all_train():
    print("let's stick all imgs together")
    s = 835

    x = np.zeros((5 * s, 5 * s, 8))
    y = np.zeros((5 * s, 5 * s, N_Cls))
    L, C1, C2 = 1.0, 6.0, 7.5
    ids = sorted(DF.ImageId.unique())
    print(len(ids))
    for i in range(5):
        for j in range(5):
            id = ids[5 * i + j]
            print(id)
            img1 = M(id)
            coastal = img1[...,0]
            image_b = img1[..., 1]
            image_g = img1[..., 2]
            image_y = img1[...,3]
            image_r = img1[..., 4]
            re = img1[..., 5]
            nir1 = img1[...,6]
            nir = img1[..., 7]
            
            evi = np.nan_to_num((nir - image_r) / (nir + C1 * image_r - C2 * image_b + L))
            evi = evi.clip(max=np.percentile(evi, 99), min=np.percentile(evi, 1))
            evi = (evi - np.min(evi))/(np.max(evi) - np.min(evi))
            evi = np.expand_dims(evi, 2)

            ndwi = (image_g - nir) / (image_g + nir)
            ndwi = (ndwi - np.min(ndwi))/(np.max(ndwi) - np.min(ndwi))
            ndwi = np.expand_dims(ndwi, 2)

            savi = (nir - image_r) / (image_r + nir)
            savi = (savi - np.min(savi))/(np.max(savi) - np.min(savi))
            savi = np.expand_dims(savi, 2)

            # binary = (ccci > 0.11).astype(np.float32) marks water fairly well
            ccci = np.nan_to_num((nir - re) / (nir + re) * (nir - image_r) / (nir + image_r))
            ccci = ccci.clip(max=np.percentile(ccci, 99.9), min=np.percentile(ccci, 0.1))
            ccci = (ccci - np.min(ccci))/(np.max(ccci) - np.min(ccci))
            ccci = np.expand_dims(ccci, 2)
            
            coastal = np.expand_dims(normalize(coastal),2)
            image_b = np.expand_dims(normalize(image_b), 2)
            image_g = np.expand_dims(normalize(image_g), 2)
            image_y = np.expand_dims(normalize(image_y), 2)
            image_r = np.expand_dims(normalize(image_r), 2)
            re = np.expand_dims(normalize(re), 2)
            nir1 = np.expand_dims(normalize(nir1), 2)
            nir = np.expand_dims(normalize(nir), 2)
            img = np.concatenate([coastal,image_b,image_g,image_y,image_r,re,nir1,nir], 2)
            
            #img = img.astype('float32')/2047
            print (img.shape, id, np.amax(img), np.amin(img))
            x[s * i:s * i + s, s * j:s * j + s, :] = img[:s, :s, :]
            for z in range(N_Cls):
                y[s * i:s * i + s, s * j:s * j + s, z] = generate_mask_for_image_and_class(
                    (img.shape[0], img.shape[1]), id, z + 1)[:s, :s]

    print (np.amax(y), np.amin(y))

    np.save('C:/Users/sebas/Documents/2021-10/Vision/VC_Project/kaggle/data/x_trn_%d' % N_Cls, x)
    np.save('C:/Users/sebas/Documents/2021-10/Vision/VC_Project/kaggle/data/y_trn_%d' % N_Cls, y)


def get_patches(img, msk, amt=10000, aug=True):
    is2 = int(1.0 * ISZ)
    xm, ym = img.shape[0] - is2, img.shape[1] - is2
    x, y = [], []
    tr = [0.4, 0.1, 0.1, 0.15, 0.3, 0.95, 0.1, 0.05, 0.001, 0.005]
    for i in range(amt):
        xc = random.randint(0, xm)
        yc = random.randint(0, ym)

        im = img[xc:xc + is2, yc:yc + is2]
        ms = msk[xc:xc + is2, yc:yc + is2]

        for j in range(N_Cls):
            sm = np.sum(ms[:, :, j])
            if 1.0 * sm / is2 ** 2 > tr[j]:
                if aug:
                    if random.uniform(0, 1) > 0.5:
                        im = im[::-1]
                        ms = ms[::-1]
                    if random.uniform(0, 1) > 0.5:
                        im = im[:, ::-1]
                        ms = ms[:, ::-1]

                x.append(im)
                y.append(ms)

    x, y = 2 * np.transpose(x, (0, 1, 2, 3)) - 1, np.transpose(y, (0, 1, 2, 3))
    print (x.shape, y.shape, np.amax(x), np.amin(x), np.amax(y), np.amin(y))
#     y = np.delete(y, 2)
#     y = np.delete(y, 3)
    return x, y


def make_val():
    print ("let's pick some samples for validation")
    img = np.load('C:/Users/sebas/Documents/2021-10/Vision/VC_Project/kaggle/data/x_trn_%d.npy' % N_Cls)
    msk = np.load('C:/Users/sebas/Documents/2021-10/Vision/VC_Project/kaggle/data/y_trn_%d.npy' % N_Cls)
    x, y = get_patches(img, msk, amt=3000)

    np.save('C:/Users/sebas/Documents/2021-10/Vision/VC_Project/kaggle/data/x_tmp_%d' % N_Cls, x)
    np.save('C:/Users/sebas/Documents/2021-10/Vision/VC_Project/kaggle/data/y_tmp_%d' % N_Cls, y)

def get_crop_shape(target, refer):
    # width, the 3rd dimension
    cw = (target.get_shape()[2] - refer.get_shape()[2])
    assert (cw >= 0)
    if cw % 2 != 0:
        cw1, cw2 = int(cw/2), int(cw/2) + 1
    else:
        cw1, cw2 = int(cw/2), int(cw/2)
    # height, the 2nd dimension
    ch = (target.get_shape()[1] - refer.get_shape()[1])
    assert (ch >= 0)
    if ch % 2 != 0:
        ch1, ch2 = int(ch/2), int(ch/2) + 1
    else:
        ch1, ch2 = int(ch/2), int(ch/2)

    return (ch1, ch2), (cw1, cw2)


def calc_jacc(model):
    img = np.load('C:/Users/sebas/Documents/2021-10/Vision/VC_Project/kaggle/data/x_tmp_%d.npy' % N_Cls)
    msk = np.load('C:/Users/sebas/Documents/2021-10/Vision/VC_Project/kaggle/data/y_tmp_%d.npy' % N_Cls)

    prd = model.predict(img, batch_size=4)
    print (prd.shape, msk.shape)
    avg, trs = [], []

    for i in range(N_Cls):
        t_msk = msk[:, i, :, :]
        t_prd = prd[:, i, :, :]
        t_msk = t_msk.reshape(msk.shape[0] * msk.shape[2], msk.shape[3])
        t_prd = t_prd.reshape(msk.shape[0] * msk.shape[2], msk.shape[3])

        m, b_tr = 0, 0
        for j in range(10):
            tr = j / 10.0
            pred_binary_mask = t_prd > tr

            jk = jaccard_score(t_msk, pred_binary_mask)
            if jk > m:
                m = jk
                b_tr = tr
        print (i, m, b_tr)
        avg.append(m)
        trs.append(b_tr)

    score = sum(avg) / 10.0
    return score, trs


def get_scalers(im_size, x_max, y_min):
    h, w = im_size  # they are flipped so that mask_for_polygons works correctly
    h, w = float(h), float(w)
    w_ = 1.0 * w * (w / (w + 1))
    h_ = 1.0 * h * (h / (h + 1))
    return w_ / x_max, h_ / y_min


        
if __name__ == '__main__':
    
    stick_all_train()
    
