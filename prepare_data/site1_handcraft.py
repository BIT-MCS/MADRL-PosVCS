import cv2 as cv
import json
import numpy as np
import sys
import os
sys.path.append(os.getcwd())
sys.path.append(os.getcwd() + '/MAPPO')
from MAPPO.env_config.F1_3_floors import config

def site1_F1(heat_positions):
    # 输入是实际坐标位置
    # 输出是图片中像素位置
    # 处理后图像大小：（1035,900,3）
    # 路网图片大小：（742, 800,3）
    try:
        floor_info_filename = './prepare_data/data/site1/F1/floor_info.json' # 这里可能要随着调整下
        with open(floor_info_filename) as f:
            floor_info = json.load(f)
        width_meter = floor_info["map_info"]["width"] # 从floor info中抠出来了
        height_meter = floor_info["map_info"]["height"] # 从floor info中抠出来了
    except Exception as e:
        floor_info_filename = './data/site1/F1/floor_info.json' # 这里可能要随着调整下
        with open(floor_info_filename) as f:
            floor_info = json.load(f)
        width_meter = floor_info["map_info"]["width"] # 从floor info中抠出来了
        height_meter = floor_info["map_info"]["height"] # 从floor info中抠出来了

    ret = []
    for pos in heat_positions:
    # 原点：（870,80）左上顶点：（180，80）右下顶点 （870，820）右上顶点（180,820）
    #img[int((height_meter - pos[1]) * 742/height_meter)][int(pos[0] * 800/width_meter)] = np.array([0,0,0])

    # 转化规则如下：
        x = int((height_meter - pos[1]) * 742/height_meter) 
        y = int(pos[0] * 800/width_meter)
        x = int(180 + 740/800 * x)
        y = int(80 + 690/742 * y)
        ret.append((x,y))   # 这里废话一下，对于确定不变的东西，写成tuple，而不是list，因为可以set of tuples 不能set of lists
    #print(ret,'done ret')
    return ret

def site1_F2(heat_positions):
    #Todo:验证是不是规则和F4一样，记得是得
    return site1_F1(heat_positions)

def site1_F3(heat_positions):
    #Todo:验证是不是规则和F4一样，记得是得
    return site1_F1(heat_positions)


def site1_F1_elevator():
    # 手动找到电梯位置在哪里，手动查找记得引用掉return 然后看图说话
    # 决策时，电梯周围一圈应该都可以有上电梯动作，这个事情要把轨迹写好 #TODO：hardcode轨迹
    return {'elevator_0':(590,610), 'elevator_1':(700,600), 'elevator_2':(830,270), 'elevator_3':(680,120),\
                'elevator_4':(550,170), 'elevator_5':(380,410)} #,'elevator_6':(510,560)}
    img = cv.imread('./prepare_data/output/mod_floorplan/site1/F3_POI_on_grid.png')
    # 从地图中选取出红色位置的点
    red = np.array([0,0,255])#  标点颜色ß
    blue = np.array([0,255,0])
    grid = []
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if (img[i][j] == blue).all() or (img[i][j] == red).all():
                grid.append((i,j))
    
    for i in range(100):
        img[380][520+i] = np.array([60,30,0])
    cv.imwrite('./F1_POI_on_grid_test.png', img)
    #return pos

def site1_F2_elevator():
    return {'elevator_0':(580,610),'elevator_1':(700,600),'elevator_2':(800,310),'elevator_3':(700,160),\
            'elevator_4':(570,160),'elevator_5':(380,410)}

def site1_F3_elevator():
    return {'elevator_0':(570,590),'elevator_1':(700,600), 'elevator_2':(810,320),'elevator_3':(700,170),\
            'elevator_4':(610,160),'elevator_5':(380,520)}


def site1_F1_map_connection():
    # 在一些特殊点上可以多加一维度动作，这个动作连接一些平时够不到的点
    return[]

def site1_F2_map_connection():
    return[]

def site1_F3_map_connection():
    return []



if __name__ == '__main__':
    site1_F1_elevator()