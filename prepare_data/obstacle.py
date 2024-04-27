import cv2 as cv
import sys
sys.path.append('/Users/haomingyang/Documents/GitHub/INDOOR/indoor_final')
import os
import numpy as np
from PIL import Image
from visualize_f import save_figure_to_png
import site2_handcraft as handcraft
from MAPPO.env_config.default_config import config


def _black_white(in_dir, out_dir):
    #将原图参差不齐的颜色全变为黑白图
    img = cv.imread(in_dir)
    for b in range(img.shape[0]):
        for g in range(img.shape[1]):
            #print(img[g][b], np.array([0,0,0]))
            if not (img[b][g] == np.array([255,255,255])).all():
                img[b][g] = np.array([0,0,0])
    ig = Image.fromarray(img)
    ig.save(out_dir)
    return ig

def _find_all_white(in_dir):
    # 显示出图中所有白色的区域
    img = cv.imread(in_dir)
    idx = []
    for b in range(img.shape[0]):
        for g in range(img.shape[1]):
            if (img[b][g] == np.array([255,255,255])).all():
                idx.append((b,g))
    return idx

def _gridline_position_on_map(in_dir, legal_pos, default_interval = config['poi_distance']):
    img = cv.imread(in_dir)
    x_split = np.arange(0,img.shape[0],default_interval) # 现在的默认间距是10
    y_split = np.arange(0,img.shape[1],default_interval) # 现在的默认间距是10
    chosen = []
    for x in x_split:
        for y in y_split:
            if (x,y) in legal_pos: # 这个速度可能比较慢，有空考虑改一下
                chosen.append((x,y))
    return chosen

def _fine_tune_modified(in_dir):
    ##把毛边处理下，让所有像素全部是白色和黑色##
    img = cv.imread(in_dir)
    for b in range(img.shape[0]):
        for g in range(img.shape[1]):
            #print(img[g][b], np.array([0,0,0]))
            if (img[b][g] > np.array([200,200,200])).any():
                img[b][g] = np.array([255,255,255])
            else:
                img[b][g] = np.array([0,0,0])
    return img
         
def _choose_color(in_dir, color):
    #选择一个图片中所有满足具体像素的点
    img = cv.imread(in_dir)
    #print(np.unique(img.reshape(-1, img.shape[-1]), axis = 0,return_counts = True))
    ret = []
    for b in range(img.shape[0]):
            for g in range(img.shape[1]):
                if (img[b][g] == color).all():
                    ret.append((b,g))
    return ret

def _visualize_gridline_position_on_map(in_dir, chosen_pos):
    #将选出的点设置为正红色
    img = cv.imread(in_dir)
    for pos in chosen_pos:
        b = pos[0]
        g = pos[1]
        img[b][g] = np.array([255,0,0]) #正红色
    return img

def _visualize_poi_on_map(in_dir, chosen_pos):
    #将选出的点设置为正绿色
    img = cv.imread(in_dir)
    #print(img.shape, chosen_pos)
    for pos in chosen_pos:
        b = pos[0]
        g = pos[1]
        img[b][g] = np.array([0,255,0]) #正绿色
    return img

def _pipeline_processing(Site,FL):
    # 读取如PS涂黑后的图片, 将图片处理成全部为[0,0,0] 和 [255,255,255]的格式并存储，命名格式FL_bb.png
    img = _fine_tune_modified(f'./output/mod_floorplan/{Site}/{FL}_mod.png')
    #print(np.unique(img.reshape(-1, img.shape[-1]), axis = 0,return_counts = True))
    ig = Image.fromarray(img)
    ig.save(f'./output/mod_floorplan/{Site}/{FL}_bb.png')
    #ig.show()

    # 读取处理后的FL_bb.png，并网格化，选取可走路径点并存储到FL_bb.png，命名为FL_env.png
    whites = _find_all_white(f'./output/mod_floorplan/{Site}/{FL}_bb.png')
    chosen = _gridline_position_on_map(f'./output/mod_floorplan/{Site}/{FL}_bb.png', whites)
    outs = _visualize_gridline_position_on_map(f'./output/mod_floorplan/{Site}/{FL}_bb.png', chosen) # 选择路径，将可走的点点上去
    #ig = Image.fromarray(outs)
    #print(np.unique(outs.reshape(-1, outs.shape[-1]), axis = 0,return_counts = True))
    #ig.save(f'./output/mod_floorplan/{Site}/{FL}_env.png')
    cv.imwrite(f'./output/mod_floorplan/{Site}/{FL}_env.png', outs)

    # 读取处理后的FL_bb.png，并网格化，选取可走路径点并存储到FL.png上，命名为FL_vis.png
    outs = _visualize_gridline_position_on_map(f'./output/mod_floorplan/{Site}/{FL}.png', chosen) # 选择路径，将可走的点点上去
    #ig = Image.fromarray(outs)
    #ig.save(f'./output/mod_floorplan/{Site}/{FL}_vis.png')
    cv.imwrite(f'./output/mod_floorplan/{Site}/{FL}_vis.png', outs)

def _pipeline_selecting(Site, FL):
    # 读取所有POI, 根据设计记录所有为正红色的点
    img = cv.imread(f'./{FL}num_AP.png')
    l = np.unique(img.reshape(-1, img.shape[-1]), axis = 0,return_counts = True)[1]
    l.tolist()
    sorted(l)
    #print(list(l))
    temp = []
    for g in range(img.shape[0]):
        for b in range(img.shape[1]):
            if (img[g][b][0] > 100 and img[g][b][1] < 100 and img[g][b][2] < 100):
                temp.append((g,b))
    #print(temp)
    cv.imwrite(f'{FL}prime.png', img)
    #ig = Image.fromarray(img)
    #ig.save(f'./{FL}prime.png')

def _find_nearest_grid(lst, grids, d = lambda x,y: (x[0]-y[0])**2 + (x[1]-y[1])**2):
    # 给定一个点列lst = [(x1,y1),(x2,y2),...],返回这些点列最近的路网坐标, 路网坐标在grids中

    ret = {grids[i]:[] for i in range(len(grids))}
    # 将像素位置放置到最近的网格上，返回格式:{(网格位置):[]}，没有match的地方是个空list
    for pos in lst:
        min_pos = grids[0]
        min_d = d(pos,min_pos)
        for cord in grids:
            l = d(pos,cord)
            if l < min_d:
                min_pos = cord
                min_d = l
        ret[min_pos].append(pos) # 加入具体位置，而不是网格位置
    
    return ret

def _gridline_matching(in_dir, match_list, width_meter, height_meter):
    # 将所有不在网格上的点，全部映射到网格上去，使用默认网格间距，每个点映射一次
    # 注意： 输入的match_list为实际坐标位置，而不是路网实际位置
    legal_whites = _find_all_white(in_dir)
    grids = _gridline_position_on_map(in_dir, legal_whites)

    # 将坐标位置转化为像素位置,这里调用handcraft module中的对应部分
    temp_converted = handcraft.site2_F4(match_list)
    converted = {tuple(temp_converted[i]): match_list[i] for i in range(len(match_list))}
    # for pos in match_list:
    #     convert_x = int(pos[0]/width_meter * 1035) # 这里用的图像默认长度 cv.imread(in_dir).shape[0]
    #     convert_y = int(pos[1]/height_meter * 900) # 这里用的图像默认长度 cv.imread(in_dir).shape[1]
    #     print(convert_x, convert_y)
    #     converted[(convert_x,convert_y)] = pos
    
    ret = _find_nearest_grid(converted.keys(), grids)
    # d = lambda x,y: (x[0]-y[0])**2 + (x[1]-y[1])**2
    # ret = {grids[i]:[] for i in range(len(grids))}
    # # 将像素位置放置到最近的网格上，返回格式:{(网格位置):[]}，没有match的地方是个空list

    # for pos in converted.keys():
    #     min_pos = grids[0]
    #     min_d = d(pos,min_pos)
    #     for cord in grids:
    #         l = d(pos,cord)
    #         if l < min_d:
    #             min_pos = cord
    #             min_d = l
    #     ret[min_pos].append(converted[pos]) # 加入具体位置，而不是网格位置

    # 将这些放到网格上的点在黑白图上点出，并生成新的图
    img = cv.imread(in_dir)
    
    for pos in ret.keys():
        if len(ret[pos]) > 0:
            img[pos[0]][pos[1]] = np.array([0,255,0])

    cv.imwrite('./test_out.png', img)

    # 将POI点返回，便于操作，返回项目{(网格位置):[（cord1), (cord2), ...]}
    return ret


if __name__ == '__main__':
    assert 'prepare_data' in os.getcwd() # 请在prepare_data 目录下运行
    _pipeline_processing('site3', 'F2')
    #_pipeline_selecting(None, 'F2')
    #_gridline_matching(f'./output/mod_floorplan/site3/F1_vis.png')

    #如果只有 Fx.png那么打开这个。
    #_black_white('./output/mod_floorplan/site3/F3.png', './output/mod_floorplan/site3/F3_mod.png')

    # #测试用，需要写成函数 看着舒服##
    # img = cv.imread('./output/mod_floorplan/site2/F2.png')
    # #img = _fine_tune_modified('./output/mod_floorplan/site2/F4_mod.png')
    # #print(np.unique(img.reshape(-1, img.shape[-1]), axis = 0,return_counts = True))
    # ig = Image.fromarray(img)
    # ig.save('./output/mod_floorplan/site2/F2_bb.png')
    # ig.show()

    # whites = _find_all_white('./output/mod_floorplan/site2/F4_bb.png')
    # chosen = _gridline_position_on_map('./output/mod_floorplan/site2/F4_bb.png', whites)
    # outs = _visualize_gridline_position_on_map('./output/mod_floorplan/site2/F4_bb.png', chosen) # 选择路径，将可走的点点上去
    # ig = Image.fromarray(outs)
    # ig.save('./output/mod_floorplan/site2/F4_env.png')
    # img = cv.imread('./data/site1/F1/floor_image.png')
    # img = cv.imread('./output/mod_floorplan/site2/F4.png')


    # for g in range(img.shape[0]):
    #     for b in range(img.shape[1]):
    #         #print(img[g][b], np.array([0,0,0]))
    #         if not (img[g][b] == np.array([255,255,255])).all():
    #             img[g][b] = np.array([0,0,0])

    # for i in range(100):  
    #     for j in range(50):          
    #         img[900+i][700+j] = np.array([100,100,100])
    # ig = Image.fromarray(img)
    # print(img.shape)
    #ig.save('./output/mod_floorplan/site2/F4_mod.png')
    #ig.show()
    #B, G, R = cv.split(img)
    #lst = []
    # print(np.unique(img.reshape(-1, img.shape[-1]), axis = 0,return_counts = True))
    # #print(img)

    # imgshow = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    # imgshow = Imag
    # e.fromarray(imgshow)
    # imgshow.show()


