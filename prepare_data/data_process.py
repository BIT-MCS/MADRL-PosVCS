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
    # 从已经标好的图片中直接读取出点的位置，也即FL_POI_on_grid.png 文件
    # 蓝色点加红色点是所有可走的路网
    # 蓝色点是没有数据的点，红色点是有数据的点
    # 返回所有带着该颜色的点的坐标
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
    # 改变时间戳
    times = int(times)/1000 
    ret = datetime.utcfromtimestamp(times).strftime('%Y-%m-%d %H:%M:%S')
    return ret

def read_txt(file_path):
    # 读取path file 文件, 仅仅筛选出 type waypoints 和 type wifi
    # 返回格式如下，会有一些ssid分开没处理的情况
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
    #也是读取path file，不过将一个folder（某一层）下的的所有文件全部读取并且整理到一个list中
    files = os.listdir(directory)
    ret = []
    for f in files:
        print(os.path.join(directory,f))
        ret.extend(read_txt(os.path.join(directory,f)))
    return ret

def second_process(lst):
    #所有路径中数据中的ssid去掉，并且将waypoint信息添加在每一个wifi测量数据的最后，并将‘TYPE_WAYPOINT"从数据集中除去
    cord = None
    ret = []
    for l in lst:
        # 先处理掉waypoints
        if l[1] == "TYPE_WAYPOINT":
            cord = (float(l[2]),float(l[3]))
            continue
        # 将cord加入每一行数据间
        temp = []
        temp.append(l[0]) # 时间
        temp.extend(l[-4:]) # bssid, rssi, frequency, last timestamp
        temp.append(cord)
        ret.append(temp)
    return ret
        
def process_sequence(directory):
    # 第一步：读取所有路径数据到一个list中
    ret = read_txt_all(directory)

    # 第二步：将所有路径中数据中的ssid去掉，并且将waypoint信息添加在每一个wifi测量数据的最后，并将‘TYPE_WAYPOINT"从数据集中除去
    # 返回例子：['1574147668484', '04:40:a9:a1:8f:c0', '-43', '5300', '1574147667752', (120.52162, 139.45494)]
    ret = second_process(ret)

    # 第三步：将所有时间戳排序,并且添加具体时间，按照事前从前到后排序，方便可视化
    sorted(ret, key = lambda x: int(x[0]))
    ret.reverse()
    tmp = []
    for seq in ret:
        l = [unix_timestamp(seq[0])]
        l.extend(seq)
        tmp.append(l)
    # 返回示例：['2019-11-19 06:14:06', '1574144046673', 'cc:08:fb:4d:e9:0d', '-91', '5785', '1574144035075', (56.791927, 105.088005)] 
    return tmp 

def all_positions(directory, outfile_name):
    # 将数据集里的点对应到可视化地图上，有坐标对不齐的问题
    # 处理思路:由于坐标无法得到，只能先将wifi数据都再写入.txt文件，然后生成成图片，进行格式转化，最后从图片的shape来对应
    #         visualize_heatmap 需要格式 {(135.24753, 145.0657): -67, (142.35493, 106.45724): -76,...}

    # 第一步：将整合好的wifi数据按照格式重新写回txt文件，给黑盒函数调用
    processed_list = process_sequence(directory)
    ret = {}
    for l in processed_list:
        if l[-1] not in ret.keys():
            ret[l[-1]] = 1
        else:
            ret[l[-1]] += 1
    # print(len(ret))
    return ret

    '''
    try:
        os.remove(outfile_name)
    except OSError:
        pass

    files = os.listdir(directory)
    write_back_read = []
    with open(outfile_name, 'w') as outfile:
        for fname in files:
            with open(os.path.join(directory,fname)) as infile:
                for line in infile:
                    outfile.write(line)
    
    return
    '''

def grid_data_extract(Site, FL, grids):
    # 将数据从文件中将数据整理到网格上
    data = process_sequence(f'./prepare_data/data/{Site}/{FL}/path_data_files/')
    pos = [d[6] for d in data] # 整合后文件的所有位置
    fc = getattr(handcraft, f'{Site}_{FL}')# 取出坐标转化函数
    pos_mapped = fc(pos) # 位置到图片位置转化
    pos_mapped_set = list(set(pos_mapped)) 
    pos_dict = obstacle._find_nearest_grid(pos_mapped_set, grids) # 找到最近的路网坐标 {路网位置：[图片位置1，图片位置2]}
    #print(pos_dict.items())
    pos_dict_reverse = {v:k for k,vs in pos_dict.items() for v in vs} # 将字典反过来，也即每一个图片坐标给最近网格
    ret = {g:[] for g in grids} 
    for i in range(len(data)): #将data的网格位置加入到list末尾
        data[i].append(pos_dict_reverse[pos_mapped[i]])
    # 返回示例：['2019-11-19 06:14:06', '1574144046673', 'cc:08:fb:4d:e9:0d', '-91', '5785', '1574144035075', (56.791927, 105.088005),(图像网格位置)] 
    for d in data: # 分下类
        ret[d[-1]].append(d)
    return ret # 考虑存储这个，格式:{网格位置:[原始数据1，原始数据2，...]}

def dict_of_grid_data(floor_list, grids_dict):
    # floor_list = [('site2','F3),('site2', 'F4),...]
    # grids_list = {('site2','F3):legal_pos, ('site2', 'F4):legal_pos,...}
    # 返回一个以楼层为key的数据字典
    return {floor_list[k]: grid_data_extract(floor_list[k][0],floor_list[k][1], grids_dict[floor_list[k]]) for k in range(len(floor_list))}

def data_time_devide(data_dict):
    # 将数据按时间分好类
    # 这个函数的主要目的是将数据按时间分分类，然后喂进data_prediction里去
    # 先确定数据集的时间长度，然后将它缩放成从早晨9点到晚上9点的这个时间段，由于timestamp是UTC时间，国内时间要快8个小时 #TODO

    # 第一步：把数据的时间统计出来
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
    # AP_amount = {}
    # for keys in data_dict.keys():
    #     AP_list = set()
    #     for items in data_dict[keys]:
    #         AP_list.add(items[2])
    #     AP_amount[keys] = len(AP_list)
    # return AP_amount

def data_prediction(data_dict, grids, AP_num = 100, RSSI_lb= -75):
    # 很明显，很多地方的数据无法采集到，需要自己补出AP的个数，我们采用的方式是拟合GP radio map，
    # 如果大于某个阈值，那么我们在该点的数据量中考虑收集这个信号
    # 输入：数据集（不同时间段或者不同信号频率这些全部由输入考虑）
    # 输出：所有位置的AP预测个数
    
    #初始化返回值，不少地方没有数据
    ret = {} #返回格式 {网格位置:AP个数，...}
    for keys in data_dict.keys():
        ret[keys] = len(data_dict[keys])

    # 第一步：挑选出AP数据量比较多的前AP_num = 200个AP，对每一个拟合一个GP
    AP_dic = {} # {AP: [map_positions]}
    for data in data_dict.values():
        for d in data:
            if d[2] not in AP_dic.keys():
                AP_dic[d[2]] = {d[-1]:[int(d[3])]} #-2 为实际坐标位置 -1 为图像网格位置  #TODO：忽略了时间,数据不应在这里处理，这里只是加权平均 
            # 输出    
            else:
                if d[-2] not in AP_dic[d[2]].keys():
                    AP_dic[d[2]][d[-1]] = [int(d[3])]
                else:
                    AP_dic[d[2]][d[-1]].append(int(d[3]))

    assert len(list(AP_dic.keys())) >  AP_num
    # AP_chosen = {k:set().add((j, sum(AP_dic[k][j])/len(AP_dic[k][j]))) for k in list(AP_dic.keys())[:AP_num] \
    # for j in AP_dic[k].keys()} # 选出AP数目前N个
    AP_dic = {k:v for k,v in sorted(AP_dic.items(), key = lambda item: len(item[1]), reverse = True)} # python3.9的dict 是有顺序的，按从大到小排序
    AP_chosen = {}
    for k in list(AP_dic.keys())[:AP_num]:
        tempset = set()
        for j in AP_dic[k].keys():
            tempset.add((j, sum(AP_dic[k][j])/len(AP_dic[k][j])))
        AP_chosen[k] = tempset

    # 第二步：构建GP，并训练一个Radio Map
    def GP_regression(AP_chosen, normalize_x = False, normalize_y= False, grid_print = True):
        # 返回 radio map, 以及AP对应的mean和标准差用于数据还原
        model = {}
        mean_std = {}
        for ap in AP_chosen.keys(): 
            # RBF sets to non-isotropic, with constant kernel to help scaling
            # as long as C is positive, Cov is PSD and hence valid kernel.

            # 大的length_scale 意味着，更平滑的函数
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
    
            #X = np.array(sorted([list(d) for d in X]))
            #y = [d for d in y] # 根据最基本GP的假设，需要把y normalize到0
            mean_std[ap] = {'X':[scalerX.mean_,scalerX.var_],'y':[scalery.mean_,scalery.var_]} # mean,std
            # #对y进行标准化
            # if normalize_y == True:
            #     meany = sum(y)/len(y)
            #     stdy = np.sqrt(sum([(item - meany)**2 for item in y])/len(y))
            #     y = np.array([(item - meany)/stdy for item in y])
            #     mean_std[ap]['y'] = [meany, stdy]
            
            # #对X进行标准化
            # if normalize_x == True:
            #     scale = StandardScaler()
            #     scale.fit(X)
            #     X = scale.transform(X)
            #     # 目前没implement标准化储存字典
            # #print(X,y)

            # 这里要考虑extrapolation的问题
            
            m.fit(X, y)
            model[ap] = m
            
            # 可视化回归的AP
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
    #print(model_dict[list(model_dict.keys())[0]].kernel_)
    #print(model_dict[list(model_dict.keys())[0]].predict(np.array([[42.256924,87.98712],[50,80],[0,0],[183.4985,98.06043]])))
    
    #第三步:根据阈值和预测的GP给所有坐标点返回是否能收到这个AP信号       
    recover = lambda rssi_list,dic,k: [rssi*dic[k]['y'][1]+ dic[k]['y'][0] for rssi in rssi_list]
    RSSI_dict = {k:recover(m.predict(grids), mean_std_dict, k) for k,m in model_dict.items()} # {AP1:[-75,-76,...], AP2:[-80,-90]}
    predict_dict = {k:[1 if RSSI_dict[k][j] > RSSI_lb else 0 for j in range(len(RSSI_dict[k]))] for k in model_dict.keys()}
    #print(predict_dict)
    return RSSI_dict, predict_dict 

def post_process_and_save_GP(RSSI_dict, predict_dict, FL, Site):
    # 由于训练GP并预测这个事情可能需要一些时间，因此我们还是对训练后的模型进行储存
    # 输入的是data_prediction中返回的dict
    pickle._dump(RSSI_dict, open(f'./intermediate/data/{Site}_{FL}_RSSIdict.pkl', 'wb'))
    pickle._dump(predict_dict, open(f'./intermediate/data/{Site}_{FL}_predictdict.pkl', 'wb'))

def load_GP(FL, Site):
    #重新读会储存的dict
    RSSI_dict = pickle.load(open(f'./intermediate/data/{Site}_{FL}_RSSIdict.pkl', 'rb'))
    predict_dict = pickle.load(open(f'./intermediate/data/{Site}_{FL}_predictdict.pkl', 'rb'))
    return RSSI_dict, predict_dict

def transition_prob_extract(data_dict):
    # 考虑无人车移动的成功概率，由于一个地方可能有人，因此有概率移动成功，这个函数从数据集中统计这个数字
    # grids 是路网legal position的集合[(x1,y1),(x2,y2),...]
    # 统计每个点的到达时间列{(x,y):[data infos]}
    for keys in data_dict:
        interval_length = {}
        for data in data_dict[keys]:
            pass

def connected_component(maps):
    # maps:[adj_matrix1, adjmatrix2,..] 按照楼层顺序排布好
    # 这个函数用来检测多少个connected components 
    # 构建的路网需要是连续的整体
    from collections import deque

    all_index = [(i,j) for i in range(len(maps)) for j in range(maps[i].shape[0])] #(fl_idx, idx)
    BFS_stack = deque()
    connected_parts = [] # 连通图[[(fl1,idx),(fl1,idx2),...],[(fl1,idx1),(fl2,idx2),..],...]
    before_length = len(all_index)

    # 检查all_index 是不是空了, all_index 只能remove
    while len(all_index) > 0:
        partial = []
        BFS_stack.append(all_index[0]) # 新的联通部分入列
        all_index.remove(all_index[0]) # 从总列中删除

        while BFS_stack:
            # 出列
            node = BFS_stack.pop() #popleft 也ok
            partial.append(node) #只在被stack吐出来的时候加
            fl = node[0]
            idx = node[1]

            # 加入他的所有邻居
            local_copy = deepcopy(all_index)
            dist_f = lambda x,y: (x[0]-y[0])**2 + (x[1]-y[1]) **2
            for pos in local_copy:
                if fl != pos[0]:
                    if maps[fl][idx][idx] == 2 and maps[pos[0]][pos[1]][pos[1]] == 2:
                        # 这个在连通处,其实所有电梯应该互相联通，这里不是问题
                        BFS_stack.append(pos)
                        all_index.remove(pos)
                elif maps[pos[0]][idx][pos[1]] == 1:
                    # idx和i是邻居且还没被任何一个联通部分算入
                    BFS_stack.append(pos)
                    all_index.remove(pos)
                    
        connected_parts.append(partial)
    print(len(connected_parts), ' parts retrived')
    assert sum([len(itm) for itm in connected_parts]) == before_length

    return connected_parts

def connect_all_parts(maps, pos_maps, bounds = 40 * 40 * 2):
    # 将一个图的几个不联通部分联通，返回一个新的联通矩阵
    # 连接逻辑：计算set distance 然后从最小的往最大的连，直到全部联通
    # bounds是平方距离 目前40 sqrt（2），即最长两端距离
    # pos_maps: {(floor_index, pos_index): (400, 250)}
    # maps: [adj1, adj2,...]

    ret_maps = deepcopy(maps)
    connected_parts = connected_component(maps)
    
    # 两两算距离
    distance_dict = {(i,j): minimum_distance_of_sets(connected_parts[i],connected_parts[j],pos_maps) \
                       for i in range(len(connected_parts)) for j in range(len(connected_parts)) if i != j}
    heaps = []
    for ele in distance_dict.items():
        heappush(heaps, (ele[1][0], ele[0])) # (distance, (i,j))
    
    connected = set()

    # 这个是用来在图里到底哪里连接了起来
    connected_list = []

    def hiting_walls(pos1, pos2, fl):
    # 这里如果黑的多于白的，那么hitting walls 就返回True
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

    # 第一步保证至少全部联通
    while len(connected) < len(connected_parts):
        # 最小距离
        print(len(heaps))
        distance,keys = heappop(heaps)
        pairs = distance_dict[keys]
        parts_a = pairs[1] #(fl_idx, pos_idx)
        parts_b = pairs[2]
        assert parts_a[0] == parts_b[0] # 这个应该要发生，因为目前不存在楼梯是任意最近点的情况
        assert maps[parts_a[0]][parts_a[1]][parts_b[1]] == 0 and maps[parts_a[0]][parts_b[1]][parts_a[1]] == 0 #这个也应该对的，证明确实没连通
        if hiting_walls(pos_maps[parts_a], pos_maps[parts_b], parts_a[0]):
            # 撞墙了就返回True
            continue
        ret_maps[parts_a[0]][parts_a[1]][parts_b[1]] = 1
        ret_maps[parts_a[0]][parts_b[1]][parts_a[1]] = 1
        connected.add(parts_a)
        connected.add(parts_b)
        connected_list.append(((parts_a[0], pos_maps[parts_a]), (parts_b[0], pos_maps[parts_b])))
    
    # 第二步保证一个范围内的都能连起来
    if len(heaps) > 0:
        next_pop = heappop(heaps)
        while next_pop[0] <= bounds:
            distance,keys = next_pop
            pairs = distance_dict[keys]
            parts_a = pairs[1]
            parts_b = pairs[2]
            assert parts_a[0] == parts_b[0] # 这个应该要发生，因为目前不存在楼梯是任意最近点的情况
            assert maps[parts_a[0]][parts_a[1]][parts_b[1]] == 0 and maps[parts_a[0]][parts_b[1]][parts_a[1]] == 0 #这个也应该对的，证明确实没连通
            if hiting_walls(pos_maps[parts_a], pos_maps[parts_b], parts_a[0]):
                # 撞墙了就返回True
                next_pop = heappop(heaps)
                continue
            ret_maps[parts_a[0]][parts_a[1]][parts_b[1]] = 1
            ret_maps[parts_a[0]][parts_b[1]][parts_a[1]] = 1
            next_pop = heappop(heaps)
            connected_list.append(((parts_a[0], pos_maps[parts_a]), (parts_b[0], pos_maps[parts_b])))

    # 将所有连接起来的地方以及连接点在图里画出
    visualize_connection(connected_list, connected_parts, pos_maps)

    return  ret_maps      

def visualize_connection(connected_list, connected, pos_maps):
    # 将联通的点画在地图上，方便观看，画在合并地图上
    # connected:(floor_index, pos_index) i.e. (0,102) 
    # pos_maps: {(floor_index, pos_index): (400, 250)}
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
    
    # 将所有联通的点标出在地图上
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
    # 计算两个set间的最小距离
    # 返回最小的距离以及对应的点
    # pa,pb: [(fl,idx),...]
    # pos_map : {(fl,idx):(580,600)} 
    #print(pos_maps)
    pa_min = pa[0]
    pb_min = pb[0]
    min_d = float('inf')
    dist_f = lambda x,y: (x[0]-y[0])**2 + (x[1]-y[1])**2
    for a in pa:
        for b in pb:
            # 由于还要考虑最短距离和连线中是否有墙的关系，因此这里的逻辑可能很case by case
            
            assert a != b
            if a[0] != b[0]:
                # 不计算不在一层的了，没意义，必然不是最小
                continue
            else:
                d = dist_f(pos_maps[a],pos_maps[b])
                if min_d > d:
                    min_d = d
                    pa_min = a
                    pb_min = b
    
    return min_d, pa_min, pb_min
            

if __name__ == "__main__":
    # 在indoor_final 路径下运行
    #F3 = process_sequence('./prepare_data/data/site3/F1/path_data_files/')
    blues,reds = grids_extract('./prepare_data/output/mod_floorplan/site2/F3_POI_on_grid.png')
    grids = [i for i in blues]
    grids.extend(reds)
    # ret = grid_data_extract('site3','F1', grids)
    # try:
    #     os.remove('./test.txt')
    # except OSError:
    #     pass
    # tools.save_dictionary('text.txt', ret)
    dic = tools.load_dictionary('text.txt')
    RSSI_dict, predict_dict = data_prediction(dic, grids)
    post_process_and_save_GP(RSSI_dict, predict_dict, 'site2', 'F3')
    RSSI_dict, predict_dict = load_GP('site2', 'F3')
    print(RSSI_dict,predict_dict)