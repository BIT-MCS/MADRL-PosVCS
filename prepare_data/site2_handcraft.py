import cv2 as cv
import json
import numpy as np
import sys
import os
sys.path.append(os.getcwd())
sys.path.append(os.getcwd() + '/MAPPO')
from MAPPO.env_config.F2_3_floors import config

# 手工对齐实际坐标位置和照片中POI的位置
# 注意这些函数没有问题！！别手贱来改 因为坐标翻转的问题所以是远点对齐再放缩的，有一点像素上的整体误差，没有大影响

def site2_F4(heat_positions):
    # 输入是实际坐标位置
    # 输出是图片中像素位置
    # 处理后图像大小：（1035,900,3）
    # 路网图片大小：（742, 800,3）
    try :
        floor_info_filename = './prepare_data/data/F4/floor_info.json' # 这里可能要随着调整下
        with open(floor_info_filename) as f:
            floor_info = json.load(f)
    except Exception as e:
        floor_info_filename = './data/F4/floor_info.json' # 这里可能要随着调整下
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

def site2_F3(heat_positions):
    #Todo:验证是不是规则和F4一样，记得是得
    return site2_F4(heat_positions)

def site2_F2(heat_positions):
    #Todo:验证是不是规则和F4一样，记得是得
    return site2_F4(heat_positions)

def site2_F5(heat_positions):
    #Todo:验证是不是规则和F4一样，记得是得
    return site2_F4(heat_positions)

def site2_F6(heat_positions):
    #Todo:验证是不是规则和F4一样，记得是得
    return site2_F4(heat_positions)

def site2_F7(heat_positions):
    #Todo:验证是不是规则和F4一样，记得是得
    return site2_F4(heat_positions)

def site2_F4_elevator():
    # 手动找到电梯位置在哪里，手动查找记得引用掉return 然后看图说话
    # 决策时，电梯周围一圈应该都可以有上电梯动作，这个事情要把轨迹写好 #TODO：hardcode轨迹
    return {'elevator_0':(620,350), 'elevator_1':(330,470), 'elevator_2':(650,530), 'elevator_3':(380,350),\
                'elevator_4':(460,310), 'elevator_5':(630,190)} #,'elevator_6':(510,560)}
    img = cv.imread('./prepare_data/output/mod_floorplan/site2/F6_POI_on_grid.png')
    # 从地图中选取出红色位置的点
    red = np.array([0,0,255])#  标点颜色
    blue = np.array([0,255,0])
    grid = []
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if (img[i][j] == blue).all() or (img[i][j] == red).all():
                grid.append((i,j))
    
    for i in range(100):
        img[580][590+i] = np.array([60,30,0])
    cv.imwrite('./F6_POI_on_grid_test.png', img)
    #return pos

def site2_F3_elevator():
    return {'elevator_0':(620,350), 'elevator_1':(330,470), 'elevator_2':(650,530), 'elevator_3':(380,350),\
            'elevator_4':(460,310), 'elevator_5':(630,270)} #'elevator_6':(570,590)}

def site2_F2_elevator():
    return {'elevator_0':(620,350), 'elevator_1':(330,470), 'elevator_2':(660,530), 'elevator_3':(380,350),\
            'elevator_4':(460,310), 'elevator_5':(630,270)} #'elevator_6':(570,590)}

def site2_F5_elevator():
    return {'elevator_0':(620,350), 'elevator_1':(330,470), 'elevator_2':(570,580), 'elevator_3':(380,350),\
        'elevator_4':(460,310), 'elevator_5':(630,270)}

def site2_F6_elevator():
    return {'elevator_0':(620,350), 'elevator_1':(330,470), 'elevator_2':(490,570), 'elevator_3':(380,350),\
        'elevator_4':(460,310), 'elevator_5':(630,270)}

def site2_F7_elevator():
    return {'elevator_0':(620,350), 'elevator_1':(330,470), 'elevator_2':(550,600),'elevator_3':(380,350),\
        'elevator_4':(460,310), 'elevator_5':(630,270)}

def site2_F4_map_connection():
    # 在一些特殊点上可以多加一维度动作，这个动作连接一些平时够不到的点
    assert config['neighbour_range'] == 2
    return [((490,430),(450,470)),((540,540),(570,510)),((490,660),(460,680)),\
           ((320,460), (290,480)),((740,210), (730, 250)),((660,320),(630,340)),((480,580),(440,610))]

def site2_F3_map_connection():
    assert config['neighbour_range'] == 2
    return [((490,430),(450,470)),((540,540),(570,510)),((490,660),(460,680)),\
           ((320,460), (290,480)),((740,210), (730, 250)),((660,320),(630,340)),((480,580),(440,610))]

def site2_F2_map_connection():
    assert config['neighbour_range'] == 2
    return [((490,430),(440,470)),((540,540),(570,510)),((500,650),(460,680)),\
           ((320,460), (290,480)),((740,190), (730, 230)),((660,320),(630,340))]

def site2_F5_map_connection():
    assert config['neighbour_range'] == 2
    return []

def site2_F6_map_connection():
    assert config['neighbour_range'] == 2
    return []

def site2_F7_map_connection():
    assert config['neighbour_range'] == 2
    return []

def site2_F4_uav_init():
    # 如果这层初始化无人机则返回这个
    return (580,600)
    img = cv.imread('./output/mod_floorplan/site2/F4_POI_on_grid.png')
    # 从地图中选取出红色位置的点
    red = np.array([0,0,255])#  标点颜色
    blue = np.array([0,255,0])
    grid = []
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if (img[i][j] == blue).all() or (img[i][j] == red).all():
                grid.append((i,j))
    
    for i in range(100):
        img[580][600+i] = np.array([60,30,0])

    cv.imwrite('./F4_POI_on_grid_test.png', img)


if __name__ == '__main__':
    site2_F4_elevator()