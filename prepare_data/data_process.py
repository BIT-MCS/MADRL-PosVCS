from datetime import datetime
import tools
import pandas as pd
import os
import main
from obstacle import *
import cv2 as cv
import site2_handcraft as handcraft
import obstacle
import pickle
import numpy as np
from heapq import *
from copy import deepcopy
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, Sum, WhiteKernel,Exponentiation
from sklearn.gaussian_process import GaussianProcessRegressor as GP
from sklearn.gaussian_process.kernels import *
from sklearn import preprocessing
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from prepare_for_rendering import _coordinate_change_after_rendering
from skimage.draw import line

 #这个文件用来将路径数据整合，判断地图上每个位置的到达时间和到达个数

def grids_extract(dir):
    blues = []
    reds = []
    img = cv.imread(dir)
    for b in range(img.shape[0]):
        for g in range(img.shape[1]): 
            if (img[b][g] == np.array([0,0,255])).all():
                blues.append((b,g))
            elif (img[b][g] == np.array([255,0,0])).all():
                reds.append((b,g))
    return blues, reds

def unix_timestamp(times):
    times = int(times)/1000 
    ret = datetime.utcfromtimestamp(times).strftime('%Y-%m-%d %H:%M:%S')
    return ret

def read_txt(file_path):
    # [['1574142027545', 'TYPE_WIFI', '04:40:a9:a1:07:a1', '-57', '5745', '1574142024318'],
    #  ['1574142027545', 'TYPE_WIFI', 'JOY', 'CITY', '04:40:a9:a1:05:60', '-63', '5180', '1574142019577']
    #  ['1574142027545', 'TYPE_WAYPOINT', 182.24911, 99.77357],...]
    ret = []
    with open(file_path, encoding = "utf-8") as fp:
        for line in fp:
            if line.split()[1] == "TYPE_WAYPOINT" or line.split()[1] == "TYPE_WIFI":
                ret.append([str(item) for item in line.split()])
    return ret

def read_txt_all(directory):
    files = os.listdir(directory)
    ret = []
    for f in files:
        print(os.path.join(directory,f))
        ret.extend(read_txt(os.path.join(directory,f)))
    return ret

def second_process(lst):
    cord = None
    ret = []
    for l in lst:
        if l[1] == "TYPE_WAYPOINT":
            cord = (float(l[2]),float(l[3]))
            continue
        temp = []
        temp.append(l[0]) 
        temp.extend(l[-4:]) # bssid, rssi, frequency, last timestamp
        temp.append(cord)
        ret.append(temp)
    return ret
        
def process_sequence(directory):
    ret = read_txt_all(directory)

    # ['1574147668484', '04:40:a9:a1:8f:c0', '-43', '5300', '1574147667752', (120.52162, 139.45494)]
    ret = second_process(ret)

    sorted(ret, key = lambda x: int(x[0]))
    ret.reverse()
    tmp = []
    for seq in ret:
        l = [unix_timestamp(seq[0])]
        l.extend(seq)
        tmp.append(l)

    return tmp 

def all_positions(directory, outfile_name):
    processed_list = process_sequence(directory)
    ret = {}
    for l in processed_list:
        if l[-1] not in ret.keys():
            ret[l[-1]] = 1
        else:
            ret[l[-1]] += 1

    return ret


def grid_data_extract(Site, FL, grids):
    data = process_sequence(f'./prepare_data/data/{FL}/path_data_files/')
    pos = [d[6] for d in data] 
    fc = getattr(handcraft, f'{Site}_{FL}')
    pos_mapped = fc(pos) 
    pos_mapped_set = list(set(pos_mapped)) 
    pos_dict = obstacle._find_nearest_grid(pos_mapped_set, grids) 
    pos_dict_reverse = {v:k for k,vs in pos_dict.items() for v in vs} 
    ret = {g:[] for g in grids} 
    for i in range(len(data)): 
        data[i].append(pos_dict_reverse[pos_mapped[i]])
    
    for d in data: 
        ret[d[-1]].append(d)
    return ret 

def dict_of_grid_data(floor_list, grids_dict):
    # floor_list = [('site2','F3),('site2', 'F4),...]
    # grids_list = {('site2','F3):legal_pos, ('site2', 'F4):legal_pos,...}
    return {floor_list[k]: grid_data_extract(floor_list[k][0],floor_list[k][1], grids_dict[floor_list[k]]) for k in range(len(floor_list))}

def data_time_devide(data_dict):
    for k,v in data_dict.items():
        if len(v) != 0:
            min_time = v[0][1]
            max_time = v[-1][1]
            for items in v:
                if items[1] < min_time:
                    min_time = items[1]
                elif items[1] > max_time:
                    max_time = items[1]
        else:
            min_time = 0
            max_time = 0
        print(unix_timestamp(min_time), unix_timestamp(max_time))

def data_prediction(data_dict, grids, AP_num = 100, RSSI_lb= -75):
    ret = {} 
    for keys in data_dict.keys():
        ret[keys] = len(data_dict[keys])

    AP_dic = {} 
    for data in data_dict.values():
        for d in data:
            if d[2] not in AP_dic.keys():
                AP_dic[d[2]] = {d[-1]:[int(d[3])]} 
            else:
                if d[-2] not in AP_dic[d[2]].keys():
                    AP_dic[d[2]][d[-1]] = [int(d[3])]
                else:
                    AP_dic[d[2]][d[-1]].append(int(d[3]))

    assert len(list(AP_dic.keys())) >  AP_num

    AP_dic = {k:v for k,v in sorted(AP_dic.items(), key = lambda item: len(item[1]), reverse = True)} 
    AP_chosen = {}
    for k in list(AP_dic.keys())[:AP_num]:
        tempset = set()
        for j in AP_dic[k].keys():
            tempset.add((j, sum(AP_dic[k][j])/len(AP_dic[k][j])))
        AP_chosen[k] = tempset

    def GP_regression(AP_chosen, normalize_x = False, normalize_y= False, grid_print = True):
        model = {}
        mean_std = {}
        for ap in AP_chosen.keys(): 
            # Up to your choice of kernel, some none smooth kernel should be better but make sure you data is enough for this GPR
            k0 =  C(1.0, (1e-3, 1e3)) * RBF(length_scale = [1,1], length_scale_bounds=(1e-3, 1e3)) # Gaussian Kernel
            k1 =  C(1.0, (1e-3, 1e3)) * Exponentiation(RBF(length_scale=[1,1],length_scale_bounds=(1e-3, 1e6)), exponent=0.5) # Laplacian Kernel
            k2 =  C(1.0, (1e-3, 1e3)) * Exponentiation(RBF(length_scale=[1,1],length_scale_bounds=(1e-3, 1e6)), exponent=0.7) # Somwhere in between, depends on your choice of smoothness
            m = GP(kernel=k0, n_restarts_optimizer=30)
            X,y = zip(*AP_chosen[ap])
            X = np.array(X)
            y = np.array(y).reshape(-1, 1)
         
            scalerX = preprocessing.StandardScaler().fit(X)
            scalery = preprocessing.StandardScaler().fit(y)
            X = scalerX.transform(X)
            y = scalery.transform(y)
    
            # Also sklearn doen't support mean function for GP regression, hence you have to normalize data
            mean_std[ap] = {'X':[scalerX.mean_,scalerX.var_],'y':[scalery.mean_,scalery.var_]} # mean,std
            
            m.fit(X, y)
            model[ap] = m
            
            # Visualization if needed
            if grid_print == True:
                X_test = np.array([[i*10, j*10] for i in range(100) for j in range(100)])
                X_test = scalerX.transform(X_test)
                y_test = m.predict(X_test)
                X_test = scalerX.inverse_transform(X_test)
                X0p, X1p = X_test[:,0],X_test[:,1]
                fig = plt.figure(figsize=(10,8))
                ax = fig.add_subplot(111, projection='3d')
                my_cmap = plt.get_cmap('plasma')
                y_test = scalery.inverse_transform(y_test.reshape(-1,1))
                y_test = [item[0] for item in y_test]
                surf = ax.plot_trisurf(np.array(X0p), np.array(X1p), np.array(y_test), linewidth=0, antialiased=False, cmap = my_cmap)
                fig.colorbar(surf, ax = ax, shrink = 0.5, aspect = 5)
                plt.show()
        return model, mean_std
    
    model_dict,mean_std_dict = GP_regression(AP_chosen, normalize_x = False, normalize_y= True, grid_print = True)   
    
    recover = lambda rssi_list,dic,k: [rssi*dic[k]['y'][1]+ dic[k]['y'][0] for rssi in rssi_list]
    RSSI_dict = {k:recover(m.predict(grids), mean_std_dict, k) for k,m in model_dict.items()} 
    predict_dict = {k:[1 if RSSI_dict[k][j] > RSSI_lb else 0 for j in range(len(RSSI_dict[k]))] for k in model_dict.keys()}
    return RSSI_dict, predict_dict 

def post_process_and_save_GP(RSSI_dict, predict_dict, FL, Site):
    pickle._dump(RSSI_dict, open(f'./intermediate/data/{Site}_{FL}_RSSIdict.pkl', 'wb'))
    pickle._dump(predict_dict, open(f'./intermediate/data/{Site}_{FL}_predictdict.pkl', 'wb'))

def load_GP(FL, Site):
    RSSI_dict = pickle.load(open(f'./intermediate/data/{Site}_{FL}_RSSIdict.pkl', 'rb'))
    predict_dict = pickle.load(open(f'./intermediate/data/{Site}_{FL}_predictdict.pkl', 'rb'))
    return RSSI_dict, predict_dict

def transition_prob_extract(data_dict):
    for keys in data_dict:
        interval_length = {}
        for data in data_dict[keys]:
            pass

def connected_component(maps):
    from collections import deque

    all_index = [(i,j) for i in range(len(maps)) for j in range(maps[i].shape[0])] #(fl_idx, idx)
    BFS_stack = deque()
    connected_parts = [] 
    before_length = len(all_index)

    while len(all_index) > 0:
        partial = []
        BFS_stack.append(all_index[0]) 
        all_index.remove(all_index[0]) 

        while BFS_stack:
            node = BFS_stack.pop() #popleft 
            partial.append(node) 
            fl = node[0]
            idx = node[1]

            local_copy = deepcopy(all_index)
            dist_f = lambda x,y: (x[0]-y[0])**2 + (x[1]-y[1]) **2
            for pos in local_copy:
                if fl != pos[0]:
                    if maps[fl][idx][idx] == 2 and maps[pos[0]][pos[1]][pos[1]] == 2:
                        BFS_stack.append(pos)
                        all_index.remove(pos)
                elif maps[pos[0]][idx][pos[1]] == 1:
                    BFS_stack.append(pos)
                    all_index.remove(pos)
                    
        connected_parts.append(partial)
    print(len(connected_parts), ' parts retrived')
    assert sum([len(itm) for itm in connected_parts]) == before_length

    return connected_parts

def connect_all_parts(maps, pos_maps, bounds = 40 * 40 * 2):
    # in case your map is not connected after pixel wise manipulation
    ret_maps = deepcopy(maps)
    connected_parts = connected_component(maps)
    
    distance_dict = {(i,j): minimum_distance_of_sets(connected_parts[i],connected_parts[j],pos_maps) \
                       for i in range(len(connected_parts)) for j in range(len(connected_parts)) if i != j}
    heaps = []
    for ele in distance_dict.items():
        heappush(heaps, (ele[1][0], ele[0])) # (distance, (i,j))
    
    connected = set()

    connected_list = []

    def hiting_walls(pos1, pos2, fl):
        discrete_line = list(zip(*line(*pos1, *pos2)))
        count_black = 0
        if fl == 0:
            fl = 'F3'
        else:
            fl = 'F4'
        imgs = cv.imread(f'./prepare_data/output/mod_floorplan/site3/{fl}_env.png')
        for points in discrete_line:
            if (imgs[points[0]][points[1]] == np.array([0,0,0])).all():
                count_black += 1
        print(count_black, len(discrete_line))
        if count_black == len(discrete_line):
            return True
        return False

    while len(connected) < len(connected_parts):
        distance,keys = heappop(heaps)
        pairs = distance_dict[keys]
        parts_a = pairs[1] 
        parts_b = pairs[2]
        assert parts_a[0] == parts_b[0] 
        assert maps[parts_a[0]][parts_a[1]][parts_b[1]] == 0 and maps[parts_a[0]][parts_b[1]][parts_a[1]] == 0 #这个也应该对的，证明确实没连通
        if hiting_walls(pos_maps[parts_a], pos_maps[parts_b], parts_a[0]):
            continue
        ret_maps[parts_a[0]][parts_a[1]][parts_b[1]] = 1
        ret_maps[parts_a[0]][parts_b[1]][parts_a[1]] = 1
        connected.add(parts_a)
        connected.add(parts_b)
        connected_list.append(((parts_a[0], pos_maps[parts_a]), (parts_b[0], pos_maps[parts_b])))
    
    if len(heaps) > 0:
        next_pop = heappop(heaps)
        while next_pop[0] <= bounds:
            distance,keys = next_pop
            pairs = distance_dict[keys]
            parts_a = pairs[1]
            parts_b = pairs[2]
            assert parts_a[0] == parts_b[0] 
            assert maps[parts_a[0]][parts_a[1]][parts_b[1]] == 0 and maps[parts_a[0]][parts_b[1]][parts_a[1]] == 0 #这个也应该对的，证明确实没连通
            if hiting_walls(pos_maps[parts_a], pos_maps[parts_b], parts_a[0]):
                next_pop = heappop(heaps)
                continue
            ret_maps[parts_a[0]][parts_a[1]][parts_b[1]] = 1
            ret_maps[parts_a[0]][parts_b[1]][parts_a[1]] = 1
            next_pop = heappop(heaps)
            connected_list.append(((parts_a[0], pos_maps[parts_a]), (parts_b[0], pos_maps[parts_b])))

    visualize_connection(connected_list, connected_parts, pos_maps)

    return  ret_maps      

def visualize_connection(connected_list, connected, pos_maps):
    in_img = cv.imread('./intermediate/pictures/concatenate_for_render.png')
    for lines in connected_list:
        start = lines[0]
        terminate = lines[1]
        if start[0] != terminate[0]:
            # 不在一层的连线，目前不可能出现
            assert True == False
        else:
            s = _coordinate_change_after_rendering(start[0],start[1],900)
            v = _coordinate_change_after_rendering(terminate[0],terminate[1],900)
            in_img = cv.line(in_img,s,v,(0, 255, 0),9)
    
    color_list = [[0,0,0],[255,0,0],[0,255,0],[0,0,255],[100,100,100],[150,150,0]]
    for i in range(len(connected)):
        color = np.array(color_list[0])
        parts = connected[i]
        for pos in parts:
            actual_pos = pos_maps[pos]
            actual_pos_transform = _coordinate_change_after_rendering(pos[0], actual_pos, 900)
            in_img[actual_pos_transform[1]][actual_pos_transform[0]] = color
                        
    cv.imwrite('./intermediate/pictures/connection.png', in_img)

def minimum_distance_of_sets(pa,pb,pos_maps):
    pa_min = pa[0]
    pb_min = pb[0]
    min_d = float('inf')
    dist_f = lambda x,y: (x[0]-y[0])**2 + (x[1]-y[1])**2
    for a in pa:
        for b in pb:
            assert a != b
            if a[0] != b[0]:
                continue
            else:
                d = dist_f(pos_maps[a],pos_maps[b])
                if min_d > d:
                    min_d = d
                    pa_min = a
                    pb_min = b
    
    return min_d, pa_min, pb_min
            

if __name__ == "__main__":
    # please run in the out most directory
    blues,reds = grids_extract('./prepare_data/output/mod_floorplan/site2/F3_POI_on_grid.png')
    grids = [i for i in blues]
    grids.extend(reds)
    dic = tools.load_dictionary('text.txt')
    RSSI_dict, predict_dict = data_prediction(dic, grids)
    post_process_and_save_GP(RSSI_dict, predict_dict, 'site2', 'F3')
    RSSI_dict, predict_dict = load_GP('site2', 'F3')
    print(RSSI_dict,predict_dict)