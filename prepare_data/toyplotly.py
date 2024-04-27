import plotly.graph_objs as go
from PIL import Image
import cv2 as cv
from visualize_f import visualize_floor_plan
from visualize_f import visualize_trajectory, visualize_heatmap, visualize_position, visualize_floor_plan, save_figure_to_html, save_figure_to_png
import numpy as np

img = cv.imread('./data/site2/F4/floor_image_prime.png')


for g in range(img.shape[0]):
    for b in range(img.shape[1]):
        if (img[g][b] != np.array([255,255,255])).all():
            img[g][b] = np.array([0,0,0])
cv.imwrite('./data/site2/F4/floor_image_black.png',img)




def find_boundaries(img):
    # 四次扫描
    l1 = list(range(img.shape[0]))
    l2 = list(range(img.shape[1]))
    l3 = [l1[-i] for i in range(1,len(l1))]
    l3.append(0)
    l4 = [l2[-i] for i in range(1,len(l2))]
    l4.append(0)
    lr_start = []
    lr_end = []
    ud_start = []
    ud_end = []

    for g in l1:
        temp_add = 0
        for b in l2:
            if (img[g][b] != np.array([0,0,0])).any():
                temp_add = b
                break
        lr_start.append(temp_add)
    
    for g in l1:
        temp_add = len(l4)
        for b in l4:
            if (img[g][b] != np.array([0,0,0])).any():
                temp_add = b
                break
        lr_end.append(temp_add)

    for b in l2:
        temp_add = 0
        for g in l1:
            if (img[g][b] != np.array([0,0,0])).any():
                temp_add = g
                break
        ud_start.append(temp_add)
    
    for b in l2:
        temp_add = len(l3)
        for g in l3:
            if (img[g][b] != np.array([0,0,0])).any():
                temp_add = g
                break
        ud_end.append(temp_add)

    #print(len(lr_start),len(lr_end),len(ud_start),len(ud_end))
    lr_range = []
    ud_range = []
    for l in range(len(lr_start)):
        pairs = (lr_start[l], lr_end[l])
        lr_range.append(pairs)

    for l in range(len(ud_start)):
        pairs = (ud_start[l], ud_end[l])
        ud_range.append(pairs)
    
    return lr_range, ud_range
        
lr_range, ud_range = find_boundaries(img)
for g in range(len(lr_range)):
    for b in range(lr_range[g][0],lr_range[g][1]):
        if (img[g][b] == np.array([0,0,0])).all():
            img[g][b] = np.array([255,255,255])

for b in range(len(ud_range)):
    for g in range(ud_range[b][0],ud_range[b][1]):
        if (img[g][b] == np.array([0,0,0])).all():
            img[g][b] = np.array([255,255,255])

for b in range(len(ud_range)):
    for g in range(0,ud_range[b][0]):
        if (img[g][b] == np.array([255,255,255])).all():
            img[g][b] = np.array([0,0,0])

for b in range(len(ud_range)):
    for g in range(ud_range[b][1],img.shape[0]):
        if (img[g][b] == np.array([255,255,255])).all():
            img[g][b] = np.array([0,0,0])

cv.imwrite('./data/site2/F4/floor_image_convexhull.png',img)




fig = visualize_floor_plan('./data/site2/F4/floor_image.png', 236.71181213998395,219.74676479990106) 
png_filename = 'llllll.png'
save_figure_to_png(fig, png_filename)
img = cv.imread('./llllll.png')
print(img.shape)
