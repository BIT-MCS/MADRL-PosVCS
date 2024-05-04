import cv2 as cv
import sys
import os
import numpy as np
from PIL import Image
from visualize_f import save_figure_to_png
import site2_handcraft as handcraft
from MAPPO.env_config.F2_3_floors import config


def _black_white(in_dir, out_dir):
    
    img = cv.imread(in_dir)
    for b in range(img.shape[0]):
        for g in range(img.shape[1]):
            
            if not (img[b][g] == np.array([255,255,255])).all():
                img[b][g] = np.array([0,0,0])
    ig = Image.fromarray(img)
    ig.save(out_dir)
    return ig

def _find_all_white(in_dir):
    
    img = cv.imread(in_dir)
    idx = []
    for b in range(img.shape[0]):
        for g in range(img.shape[1]):
            if (img[b][g] == np.array([255,255,255])).all():
                idx.append((b,g))
    return idx

def _gridline_position_on_map(in_dir, legal_pos, default_interval = config['poi_distance']):
    img = cv.imread(in_dir)
    x_split = np.arange(0,img.shape[0],default_interval) 
    y_split = np.arange(0,img.shape[1],default_interval) 
    chosen = []
    for x in x_split:
        for y in y_split:
            if (x,y) in legal_pos: 
                chosen.append((x,y))
    return chosen

def _fine_tune_modified(in_dir):
    
    img = cv.imread(in_dir)
    for b in range(img.shape[0]):
        for g in range(img.shape[1]):
            
            if (img[b][g] > np.array([200,200,200])).any():
                img[b][g] = np.array([255,255,255])
            else:
                img[b][g] = np.array([0,0,0])
    return img
         
def _choose_color(in_dir, color):
    
    img = cv.imread(in_dir)
    
    ret = []
    for b in range(img.shape[0]):
            for g in range(img.shape[1]):
                if (img[b][g] == color).all():
                    ret.append((b,g))
    return ret

def _visualize_gridline_position_on_map(in_dir, chosen_pos):
    
    img = cv.imread(in_dir)
    for pos in chosen_pos:
        b = pos[0]
        g = pos[1]
        img[b][g] = np.array([255,0,0]) 
    return img

def _visualize_poi_on_map(in_dir, chosen_pos):
    
    img = cv.imread(in_dir)
    
    for pos in chosen_pos:
        b = pos[0]
        g = pos[1]
        img[b][g] = np.array([0,255,0]) 
    return img

def _pipeline_processing(Site,FL):
    
    img = _fine_tune_modified(f'./output/mod_floorplan/{Site}/{FL}_mod.png')
    
    ig = Image.fromarray(img)
    ig.save(f'./output/mod_floorplan/{Site}/{FL}_bb.png')
  
    whites = _find_all_white(f'./output/mod_floorplan/{Site}/{FL}_bb.png')
    chosen = _gridline_position_on_map(f'./output/mod_floorplan/{Site}/{FL}_bb.png', whites)
    outs = _visualize_gridline_position_on_map(f'./output/mod_floorplan/{Site}/{FL}_bb.png', chosen) 

    cv.imwrite(f'./output/mod_floorplan/{Site}/{FL}_env.png', outs)

    outs = _visualize_gridline_position_on_map(f'./output/mod_floorplan/{Site}/{FL}.png', chosen) 

    cv.imwrite(f'./output/mod_floorplan/{Site}/{FL}_vis.png', outs)

def _pipeline_selecting(Site, FL):
    
    img = cv.imread(f'./{FL}num_AP.png')
    l = np.unique(img.reshape(-1, img.shape[-1]), axis = 0,return_counts = True)[1]
    l.tolist()
    sorted(l)
    
    temp = []
    for g in range(img.shape[0]):
        for b in range(img.shape[1]):
            if (img[g][b][0] > 100 and img[g][b][1] < 100 and img[g][b][2] < 100):
                temp.append((g,b))
    
    cv.imwrite(f'{FL}prime.png', img)
    
    

def _find_nearest_grid(lst, grids, d = lambda x,y: (x[0]-y[0])**2 + (x[1]-y[1])**2):
    ret = {grids[i]:[] for i in range(len(grids))}
    for pos in lst:
        min_pos = grids[0]
        min_d = d(pos,min_pos)
        for cord in grids:
            l = d(pos,cord)
            if l < min_d:
                min_pos = cord
                min_d = l
        ret[min_pos].append(pos) 
    
    return ret

def _gridline_matching(in_dir, match_list, width_meter, height_meter):
     
    legal_whites = _find_all_white(in_dir)
    grids = _gridline_position_on_map(in_dir, legal_whites)

    temp_converted = handcraft.site2_F4(match_list)
    converted = {tuple(temp_converted[i]): match_list[i] for i in range(len(match_list))}

    ret = _find_nearest_grid(converted.keys(), grids)

    img = cv.imread(in_dir)
    
    for pos in ret.keys():
        if len(ret[pos]) > 0:
            img[pos[0]][pos[1]] = np.array([0,255,0])

    cv.imwrite('./test_out.png', img)
    return ret


if __name__ == '__main__':
    assert 'prepare_data' in os.getcwd() 
    _pipeline_processing('site3', 'F2')
    
    

    
    

    
    
    
    
    
    
    

    
    
    
    
    
    
    


    
    
    
    
    

    
    
    
    
    
    
    
    
    
    
    

    
    
    
    


