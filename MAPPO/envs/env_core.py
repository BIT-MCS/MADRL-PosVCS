import json
import os
import argparse
import pickle
from pathlib import Path
import numpy as np
import cv2 as cv
from PIL import Image
from copy import deepcopy
import sys

#在indoor_final 界面运行
sys.path.append(os.getcwd())
sys.path.append(os.getcwd() + '/MAPPO')
sys.path.append(os.getcwd() + '/prepare_data')
sys.path.append(os.getcwd() + '/MI_intrinsic')
from obstacle import _choose_color
from compute_f import split_ts_seq, compute_step_positions
from io_f import read_data_file
from visualize_f import visualize_trajectory, visualize_heatmap, visualize_floor_plan, save_figure_to_html, save_figure_to_png
from obstacle import *
from obstacle import _visualize_poi_on_map
from data_process import dict_of_grid_data, data_prediction,load_GP, post_process_and_save_GP,connect_all_parts,connected_component
from MAPPO.env_config import F2_3_floors #Site3_3_floors
import site2_handcraft as handcraft
import cvxpy as cp
from skimage.draw import line
import VIME_based_discovery as VI
from statistics import median
import torch


class EnvCore(object):
    """
    # 环境中的智能体
    """
    def __init__(self):
        # 环境信息储存在env_config文件夹中
        #self.config = default_config.config
        self.config = F2_3_floors.config #Site3_3_floors.config 
        if self.config['if_read_states_besides_config']:
            flag = True
        
        #如果读取
        if self.config['if_read_states']:
            with open('./intermediate/data/states_info.pkl', 'rb') as f:
                myinstance = pickle.load(f)

            for k in myinstance.__dict__.keys():
                setattr(self, k, getattr(myinstance, k))
            if flag:
                # 除了config外的信息load
                #self.config = default_config.config
                self.config = F2_3_floors.config #Site3_3_floors
            print('States Info Loaded.')
            self.total_data = self._init_data_amount() # 环境里的数据总量，目前是三次一样的数据，不一样的时候记得更新 _compute_total_data
            self.runtime_data = 0
            self.aoi_violated_ratio = 0 # TODO:这个统计量计算了数据点按时采集率,目前只在runtime中有用，不需要reset

        else:
            self.agent_num = self.config['agent_num']  # 设置智能体(小飞机)的个数，这里设置为两个
            #self.obs_dim =   # 设置智能体的观测纬度
            self.malicious_flag = [False for _ in range(self.agent_num)] #环境记录是否有malicious
            self.action_dim = self.config['total_action_dim'] # 设置智能体的动作纬度，这里假定为一个五个纬度的 #TODO:config
            self.game_steps = 0
            self.horizon = self.config['horizon']
            self._state_init()
            self._init_aoi()
            self._init_elevator_reward()
            
            # 7-1: 尽量合并到环境
            # if self.config['use_vime']:
            #     # vime 不reset, 只reset obs
            #     self._init_vime()
        
            self.total_data = self._init_data_amount() # 环境里的数据总量，目前是三次一样的数据，不一样的时候记得更新 _compute_total_data
            self.runtime_data = 0
            self.aoi_violated_ratio = 0 # TODO:这个统计量计算了数据点按时采集率,目前只在runtime中有用，不需要reset
        

            # 如果使用对偶梯度方式，初始化lambda
            # WARNING 对偶梯度在reset时候不重置，别手贱
            if self.config['use_dual_descent']:
                # TODO: 应该训练完定期和模型一起存一下lambda，方便续杯
                self._init_lambda_aoi()

            # 记录一些从环境建立到结束一直有的信息，这些信息不重置

            # 记录所有访问次数，用于看探索情况
            # 格式{('site2','F3'):{(200,100):{0:100, 1:200,...},...},...}
            self.visit_counts = {fl:{pos:{i:0 for i in range(self.agent_num)} for pos \
                    in self.legal_pos[fl]} for fl in self.config['floor_list']}

            try:
                os.remove('./intermediate/data/states_info.pkl')
            except OSError:
                pass
            with open('./intermediate/data/states_info.pkl', 'wb') as f:
                pickle.dump(self, f)
            print('States Info Saved.')


    def reset(self):
        """
        # self.agent_num设定为2个智能体时，返回值为一个list，每个list里面为一个shape = (self.obs_dim, )的观测数据
        """
        # observation 设置为初始， step归0
        
        sub_agent_obs = []
        self.uav_pos = [self.config['uav_init_pos'] if i < int(self.config['agent_num']/2) else self.config['uav_init_pos_2'] for i in range(self.config['agent_num'])] # 直接重置位置
        self.game_steps = 0
        self.data_amount = deepcopy(self.data_amount_init)
        # 3-23加入随机初始化数据训练, init不能动！！！
        self._data_amount_true_env()
        self._init_aoi() # 初始化aoi
        self.first_step = [self.agent_num for _ in range(self.agent_num)] # 上楼梯重置
        self.elevator_usage_time = [0 for _ in range(self.agent_num)] # 对于每个无人车的上电梯记录
        self.elevator_lock = np.zeros((len(self.elevator[list(self.elevator.keys())[0]]), int(self.config['data_changes_num']),\
                int(self.config['floor_num']))) # 对于每个电梯上电梯的记录锁[[[1,1,0],[0,1,0]],...]
        self.trajectory_median = [[] for _ in range(self.agent_num)] # VIME的median在trajectory结束后重置
        
        #6-29：occupancy 的更新
        # TODO：第一次更新这个要报错
        # for fl in self.config['floor_list']:
        #     print('before updates', list(self.last_occupancy[fl].values()))
        self.last_occupancy = deepcopy(self.current_occupancy)
        # for fl in self.config['floor_list']:
        #     print('after updates', list(self.last_occupancy[fl].values()))
        self.current_occupancy = {fl:{pos:self.config['agent_num'] - 1 for pos in self.poi[fl]} for fl in self.config['floor_list']} #{fl1:{(100,200): uav_index}}

        # if self.config['use_vime']:
        #     self._reset_vime()


        for i in range(self.agent_num):
            sub_obs = self._find_observations(i)
            sub_agent_obs.append(sub_obs)
        return sub_agent_obs

    def step(self, actions, rend = False):
        """
        # self.agent_num设定为2个智能体时，actions的输入为一个2纬的list，每个list里面为一个shape = (self.action_dim, )的动作数据
        # 默认参数情况下，输入为一个list，里面含有两个元素，因为动作纬度为5，所里每个元素shape = (5, )
        """
        # step里完成s,a,s'这个过程
        # actions 格式：[[0,0,0,1,0,0,...],[0,1,..],..]
        '''
        这里remind下什么参数，需要在一次step之后更改
        1. self.data_amount
        2. self.game_steps
        3. self.uav_pos
        '''
        
        sub_agent_obs = []
        sub_agent_reward = []
        sub_agent_done = []
        sub_agent_info = []
        self.game_steps += 1

        # step动作让aoi自动更新, 如果不用aoi可以去掉节约时间
        self._aoi_default_updates()

        # 是否更新随时间数据
        if self.config['use_changing_data']:
            if self.config['use_same_data'] and self.game_steps % self.config['data_last_time'] == 0:
                # TODO: 目前是到点了就全部给份新的
                self.data_amount = deepcopy(self.data_amount_init)                

        for i in range(self.agent_num):
            #先对环境进行更新
            action_selected = list(actions[i]).index(1)
            transformed_action = self._action_1_2_transform(action_selected, usage = '1_to_2')
            # 4-1:这里整合了一下aoi和数据的更新
            delta_data_amount, aoi_delta, visit_pos_list = self._data_change(i,transformed_action) # 数据量的变化，作为目前的奖励
            self.runtime_data +=  delta_data_amount
            # info有个基本作用，那就是传递available actions
            info_dict = {}
            if rend:
                # 方便別的算法用環境
                self.take_out_dict = {}

            if self._validate_action(action_selected, self.uav_pos[i]):
                if rend and self.config['use_malicious_evaluate']:
                    if self.data_amount[self.uav_pos[i][0]][self.uav_pos[i][1]] > 0:
                        transformed_action = (0,0)
                        # maliciou flag后续用在环境中生存观测
                        self.malicious_flag[i] = True
                        r = 1 #真实的奖励，只有到了才给个1
                    else:
                        # 在这个环境里，加入转移概率后，需要先考虑是否能转移成功
                        p_tran = self.transition_prob[self.uav_pos[i][0]][self.uav_pos[i][1]]
                        tuple_list = [transformed_action, (0,0)]
                        idx = np.random.choice([0,1], 1, \
                                        p = [p_tran, 1 - p_tran])[0]
                        true_action = tuple_list[idx]
                        r = 0

                elif self.config['malicious_exist']:
                    # 目前简单版的malicious，恶毒程度拉满
                    if  self.malicious_flag[i] == False and delta_data_amount > 0:
                        transformed_action = (0,0)
                        #环境标记malicious，用来生成obs
                        self.malicious_flag[i] = True
                        r = self._reward_compute(i,transformed_action,delta_data_amount)    
                    else:
                        self.malicious_flag[i] = False
                        r = -10
                       
                else:
                    # 在这个环境里，加入转移概率后，需要先考虑是否能转移成功
                    p_tran = self.transition_prob[self.uav_pos[i][0]][self.uav_pos[i][1]]
                    tuple_list = [transformed_action, (0,0)]
                    idx = np.random.choice([0,1], 1, \
                                    p = [p_tran, 1 - p_tran])[0]
                    true_action = tuple_list[idx]
                    r = self._reward_compute(i,true_action,delta_data_amount)

                # 如果dual descent，需要这里更改奖励为lagrangian奖励
                # constraint奖励分到了无人机上，如果同时到一个states根据目前更新顺序，序号小的uav得
                # if self.config['use_dual_descent'] and not rend:
                #     r = self._constraints_aoi_reward(r, i, visit_pos_list)
                sub_agent_reward.append([r])

                # 转移成功，应该在最后
                self._update_pos(i,transformed_action)
            else: 
                # 转移失败
                #TODO: 这里其实永远不该进来的
                #print('failed action transform ', i , ' ',action_selected, info_dict['available_actions'])
                print('failed action transformed ',self.game_steps)
                sub_agent_reward.append([0])

            # mask掉不去的动作
            if self.config['use_masked_action']:
                info_dict['available_actions'] = self.masked_action[self.uav_pos[i][0]][self.uav_pos[i][1]]

            # 更新AOI
            # 4-1 这部分已经放到了数据变化中
            #delta = self._aoi_updates(transformed_action, i)

            # 由于lambda要在整个episode结束后更新
            if self.config['use_dual_descent']:
                info_dict['aoi_reduce'] = aoi_delta

            # 转移完成后，加入观测

            sub_agent_obs.append(self._find_observations(i)) # 加入观测，编号好依次加入
        

            if self._terminate_condition(): # game_step > horizon 返回 True
                sub_agent_done.append(True) # 结束条件
            else:
                sub_agent_done.append(False)
            
            #只在render的时候开启
            if rend:
                keys = ['current_position' ,'transformed_action','pos_info','available_actions', 'collection_ratio', 'violation_ratio']
                pos_dict = {'pos':{fl:[pos for pos in self.legal_pos[fl]] for fl in self.config['floor_list']},
                        'val':{fl:[self.data_amount[fl][pos] for pos in self.legal_pos[fl]] for fl in self.config['floor_list']}}
                info_dict = {keys[0]: self.uav_pos[i], keys[1]:transformed_action, keys[2]:pos_dict}
                info_dict[keys[3]] = self.masked_action[self.uav_pos[i][0]][self.uav_pos[i][1]]
                info_dict[keys[4]] = self.runtime_data / self.total_data
                info_dict[keys[5]] = self._compute_aoi_violation()

            sub_agent_info.append(info_dict) # individual_reward usage 
            
            # 最后记录一些全局不变更信息，不用的话可以去掉减少时间

            # 给无人机i更新新的访问次数
            self.visit_counts[self.uav_pos[i][0]][self.uav_pos[i][1]][i] += 1
        if rend:
            self.take_out_dict = deepcopy(info_dict)

        # 6-30: 在这里添vime更新信息
        # if self._terminate_condition() and self.config['use_vime']:
        #     import time
        #     rp = self._vime_update_rewards(self.current_observation_history, sub_agent_info[0]['available_actions'],sub_agent_done,[])
        #6-29: 这里发现多reset了一次 TODO：没找到哪里reset的 靠print发现码自带reset
        
        # if self._terminate_condition(): 
        #     self.reset() #都操作完了再reset
        
        # 7-1： 这里加入s到self，
        
        #self.current_observation_history.append(sub_agent_obs)
        # 返回用于训练转移概率的data
        # sub_agent_vime = self._construct_transition_observation(i,)
        return [sub_agent_obs, sub_agent_reward, sub_agent_done, sub_agent_info]

        '''
        self.env = config
        self._generate_poi()
        self._generate_uavs() 
        '''
    
    def _runtime_summary(self):
        # 这个函数负责打印一些runtime 信息
        
        # 统计runtime时的访问次数 {100次访问的:2个点，...}
        ret = {}
        for fl in self.config['floor_list']:
            for pos in self.visit_counts[fl]:
                total_visit = sum(list(self.visit_counts[fl][pos].values()))
                if total_visit not in ret:
                    ret[total_visit] = 1
                else:
                    ret[total_visit] += 1
        
        print('visit counts, form {visits: counts of visits} ', dict(sorted(ret.items())))

    def _state_init(self):
        # 输入：楼层POI图片路径，依次给出，index 0为最下层，依次连接；
        # picture_dir_dict: {('site2','F3'), ('site2', 'F4),...,}, 例如，其中路径1代表F2，路径2代表F3，以此类推
        # 任务：将states初始化到环境中。

        # 第一步：确定路网结构以及POI
        self._init_picture() #读入地图,记录下地图的信息
        self._init_poi() # 对于每层，都初始化所有POI位置self.poi，以及路网位置self.legal_pos
        self._adjacency_matrix() # 用连接矩阵表示
        self._data_amount_extract(dict_of_grid_data(self.config['floor_list'], self.legal_pos)) # 数据量单独定义，需要根据需求来弄
        self._init_transition() # 转移方程，如果考虑人流 需要一个转移概率，如果有人则移动失败，人的出现采样完后hardcode进去
        self.elevator_transition_prob = {}# 在电梯外：没人等电梯时的概率，有人等电梯时候的概率。在电梯里：到达计划楼层的概率
        self.masked_action = self._masked_action() # load 进action_dict,如果本地没有会重新生成

        # 第二步：初始化UAV的位置以及观测
        self._init_uav()
        self.initial_obs = [self._find_observations(i) for i in range(self.config['agent_num'])]
        self.obs_dim = len(self.initial_obs[0])

        # 第三步：初始化一些和奖励有关的额外环境信息
        self._init_aoi() # 初始化aoi
        self.first_step = [self.agent_num for _ in range(self.agent_num)] # 用来记录是不是第一次上下楼
        self.elevator_usage_time = [0 for _ in range(self.agent_num)]
        self.elevator_lock = np.zeros((len(self.elevator[list(self.elevator.keys())[0]]), int(self.config['data_changes_num']),\
                                      int(self.config['floor_num'])))

        # 第四步：初始化VIME
        self.trajectory_median = [[] for _ in range(self.agent_num)]
    
    def _init_picture(self):
        # 将楼层的各种图片储存在这个字典中 {('site','floor'): [mapdir1, mapdir2,...]}
        self.picture_dict = {k:[] for k in self.config['floor_list']}
        self.black_array = {k:[] for k in self.config['floor_list']}
        for keys in self.picture_dict:
            site = keys[0]
            floor = keys[1]
            #图池，后边可以继续添加
            maplist = [
                # TODO：这里有一个对应关系，当map多了要记录下
                f'./prepare_data/output/mod_floorplan/{site}/{floor}_POI_on_grid.png' # 这个图点出了所有POI还有可以走的路径，
            ]
            #障碍池，后边可以继续加,直接读出array存储
            im = cv.imread(f'./prepare_data/output/mod_floorplan/{site}/{floor}_env.png')
            self.black_array[keys] = im
            self.picture_dict[keys].extend(maplist)
        self.x = {k:cv.imread(self.picture_dict[k][0]).shape[0] for k in self.picture_dict.keys()} # 图像的横轴
        self.y = {k:cv.imread(self.picture_dict[k][0]).shape[1] for k in self.picture_dict.keys()} # 图像的纵轴
    
    def _init_transition(self):
        # 原始环境中的转移概率等于碰撞概率
        # 在我们的环境中转移只有原地和成功两种可能
        self.transition_prob = {fl:{pos:0 for pos in self.legal_pos[fl]} for fl in self.config['floor_list']}
        
        if self.config['transition_mode'] == 'full_success':
            # 转移成功概率为1的时候
            for fl in self.config['floor_list']:
                for pos in self.legal_pos[fl]:
                    self.transition_prob[fl][pos] = 1
        elif self.config['transition_mode'] == 'with_random_collision':
            np.random.seed(100)
            # 控制在0.7-1之间
            problist = 1 - 0.3 * np.random.rand(sum([len(self.legal_pos[fl]) for fl in self.config['floor_list']]))
            count = 0
            for fl in self.config['floor_list']:
                for pos in self.legal_pos[fl]:
                    self.transition_prob[fl][pos] = problist[count]
                    count += 1
        elif self.config['transition_mode'] == 'with_deterministic_collision':
            pass
        elif self.config['transition_mode'] == 'with_malicious_and_deterministic_collision':
            # 目前的自带转移概率为1
            pass
            
    def _init_poi(self):
        # 因为有多层楼的关系，我们的输入为dict {('site2','F3'):[dir], ('site2','F3'):[dir]}
        self.poi = {k: self._load_red_from_maps(v[0]) for k,v in self.picture_dict.items()}
        self.legal_pos = {k: self._load_blue_from_maps(v[0]) + self._load_red_from_maps(v[0]) for k,v in self.picture_dict.items()}
        l = sum([len(self.poi[k]) for k in list(self.poi.keys())])
        print('Total ' + str(l) +' poi')

    def _load_red_from_maps(self, in_dir):
        # 从地图中选取出红色位置的点
        img = cv.imread(in_dir)
        red = np.array([0,0,255])#  标点颜色
        pos = _choose_color(in_dir, red)
        return pos

    def _load_blue_from_maps(self, in_dir):
        # 从地图中选取出蓝色位置的点
        img = cv.imread(in_dir)
        blue = np.array([255,0,0])#  标点颜色
        pos = _choose_color(in_dir, blue)
        return pos
    
    def _adjacency_matrix(self):

        # 第一步：首先检索并存储电梯位置 格式：{('site2', 'F3'): {'elevator_1': (620, 350), 'elevator_2': (330, 470)}, ...},...}
        self.elevator = {k:getattr(handcraft, f'{k[0]}_{k[1]}_elevator')() for k in self.config['floor_list']}
        
        # 第二步：为每层建立连接矩阵，有3x3和5x5的两种链接矩阵，先用5x5的
        def local_adjacency(pos, max_x, max_y, distance, fl):
            # distance 代表图片中等距采样的间隔大小，也即np.arrange(0,max_x,distance)
            # max.x代表cv.imread出来后的矩阵的第一项， max.y代表cv.imread出来后矩阵的第二项
            # 连结矩阵 A[i,j] = 1 代表pos[i]和pos[j]之间按相邻，可以一步走到
            adj = np.zeros([len(pos), len(pos)])
            # 位置矩阵 A[i,j] = 1 代表(i,j)这个位置在pos中出现了，这么做为了方便查表，减少搜索时间
            index_matrix = np.zeros([max_x, max_y]) # max_x 是img.shape[0] max_y是img.shape[1]
            for p in pos:
                index_matrix[p[0]][p[1]] = 1

            # 填充连接矩阵 
            for x in range(max_x):
                for y in range(max_y):
                    #如果不是1直接跳过
                    if index_matrix[x][y] != 1:
                        continue
                    #如果是1，判断周围一圈（目前距离）0是否在index_matrix 内
                    idx1 = pos.index((x,y))
                    for i in range(self.config['neighbour_range'] * 2 + 1): # 3x3用3 5x5用5
                        for j in range(self.config['neighbour_range'] * 2 + 1): # 3x3用3 5x5用5
                            # 确保没有index out of range后这个点确实也在pos内
                            
                            #TODO:3-15 这里更新，把1变为self.config['neighbour_range']，注意检查
                            # 3-23: 暂且改回可以撞墙
                            x1 = x + (i-self.config['neighbour_range']) * distance
                            y1 = y + (j-self.config['neighbour_range']) * distance
                            if x1 in range(max_x) and y1 in range(max_y) and index_matrix[x1][y1] == 1:
                                #if not self._check_hiting_walls((x,y),(x1,y1),fl):
                                if True:    
                                    idx2 = pos.index((x1, y1))
                                    adj[idx1][idx2] = 1
                                    #adj[idx2][idx1] = 1
            #返回连接矩阵，用来表示一张图
            #TODO:用5x5的可能会有穿模,可视化上后期可以细调

            assert (adj.T == adj).all()

            return adj 
        
        # 先给每层建立连接矩阵
        self.fl_map = {}
        for keys in self.config['floor_list']:
            self.fl_map[keys] = local_adjacency(self.legal_pos[keys],self.x[keys],self.y[keys], self.config['poi_distance'], keys) 
        
        # 再将电梯处的连接加上，连接矩阵处要是可以选择上楼，则地图处标记为2
        # 还需要拿index对齐

        # 把能上电梯位置和电梯对应关系的mapping存下来
        self.ele_pos_mapping = {fl:{} for fl in self.fl_map.keys()} # pos -> ele 的keys 即 elevator_1..

        # TODO:这里目前默认了电梯位置一致
        # 3-25 电梯位置允许不一致了
        for fl in self.fl_map.keys():
            for el in self.elevator[fl].values():
                ele_index = self.legal_pos[fl].index((el[0], el[1]))
                self.fl_map[fl][ele_index][ele_index] = 2
                # 电梯周围的点都可以选择上电梯这个操作，也记为2
                for i in range(-self.config['up_down_range'],self.config['up_down_range']):
                    for j in range(-self.config['up_down_range'],self.config['up_down_range']):
                        pos = (el[0] + i * self.config['poi_distance'], 
                          el[1] + j * self.config['poi_distance'])
                        if pos in self.legal_pos[fl]:    
                            self.ele_pos_mapping[fl][pos] = list(self.elevator[fl].keys())[list(self.elevator[fl].values()).index(el)]
                            idx = self.legal_pos[fl].index(pos)
                            self.fl_map[fl][idx][idx] = 2
            # 必须是对称矩阵
            assert (self.fl_map[fl].T == self.fl_map[fl]).all()

        #self._make_adjacency_connected()
        self.super_lists = {k:getattr(handcraft, f'{k[0]}_{k[1]}_map_connection')() for k in self.config['floor_list']}
        self.super_lists_split = {fl: [pos for items in self.super_lists[fl] for pos in items] for fl in self.config['floor_list']}
        for fl in self.config['floor_list']:
            for connect_pairs in self.super_lists[fl]:
                assert connect_pairs[0] in self.legal_pos[fl] and connect_pairs[1] in self.legal_pos[fl]
                idx1 = self.legal_pos[fl].index(connect_pairs[0])
                idx2 = self.legal_pos[fl].index(connect_pairs[1])
                self.fl_map[fl][idx1][idx2] = 1
                self.fl_map[fl][idx2][idx1] = 1
            assert (self.fl_map[fl].T == self.fl_map[fl]).all()
            
    def _make_adjacency_connected(self):
        # 将地图全连接了
        # 将连接矩阵弄成全部连接好的格式
        in_maps = []
        in_pos_dict = {}
        for fl_idx in range(len(self.config['floor_list'])):
            in_maps.append(self.fl_map[self.config['floor_list'][fl_idx]])
            for i in range(len(self.legal_pos[self.config['floor_list'][fl_idx]])):
                in_pos_dict[(fl_idx, i)] = self.legal_pos[self.config['floor_list'][fl_idx]][i]
        
        ret_list = connect_all_parts(in_maps, in_pos_dict)
        assert len(connected_component(ret_list)) == 1
        new_fl_dict = {}
        for fl_idx in range(len(self.config['floor_list'])):
            # 检测对称矩阵
            assert (ret_list[fl_idx].T == ret_list[fl_idx]).all()
            new_fl_dict[self.config['floor_list'][fl_idx]] = ret_list[fl_idx]
        self.fl_map = new_fl_dict
          
    def _check_hiting_walls(self, pos1, pos2, fl):
        # 撞墙动作必须被处理掉，干脆在动作空间里就干掉
        # 使用全黑图 F3_env.png，判断连线上是否有黑的
        # pos1, pos2 这里要用实际图中位置（700,800）（710,810）
        # True 是撞墙了 False是没撞墙
        discrete_line = list(zip(*line(*pos1, *pos2)))
        for points in discrete_line:
            if (self.black_array[fl][points[0]][points[1]] == np.array([0,0,0])).all():
                return True
        return False
    
    def _data_amount_extract(self, data_dict, save_dict = False):
        # 为每一层每一个AP建立一个radio map
        # data_dict 在原有的dict基础上外边再套了一层dict 即{(site,flr):{'pos1':[data1, data2,..], 'pos2':[...]},...}
        self.data_amount = {k:{kp:0 for kp in self.legal_pos[k]} for k in data_dict.keys()}
        self.data_amount_init = {k:{kp:0 for kp in self.legal_pos[k]} for k in data_dict.keys()} # 备份一下，这个不改，方便重置
        self.all_poi_num = sum([len(self.legal_pos[fs]) for fs in data_dict.keys()])
        total_amount = 0
        for fl in self.config['floor_list']:
            try :
                print(f'Load GP model at {fl[0]} {fl[1]} for data amount prediction\n')
                RSSI_dict, predict_dict = load_GP(fl[0], fl[1])
            except FileNotFoundError:
                print('Generating New GP at {fl[0]} {fl[1]} with default settings\n')
                save_dict = True # 储存dict
                RSSI_dict, predict_dict = data_prediction(data_dict[fl], self.legal_pos[fl]) #输出的顺序和输入顺序对着的，即和legal_pos对着的
            if save_dict:
                print('A new model at {fl[0]} {fl[1]} is saved to current directory\n')
                post_process_and_save_GP(RSSI_dict, predict_dict, fl[0], fl[1]) 

            #根据predict_dict计算数据量, 仅给poi数据量, 不是所有legal_pos!!!
            # RSSI_dict :{'rssi1':[-78,-77,...],'rssi2':[-67,...],...}
            
            for ks in predict_dict.keys():
                for i in range(len(self.poi[fl])):
                    # 这个pos还带着间距
                    pos = self.poi[fl][i]
                    idx = self.legal_pos[fl].index(pos)
                    #print(self.data_amount)
                    if predict_dict[ks][idx] == 1:
                        total_amount +=1
                        self.data_amount[fl][pos] += 1
                        self.data_amount_init[fl][pos] += 1
        print('Total amount of data: ' + str(total_amount))
     ########
    
    def _data_amount_true_env(self):
        # 这里用实际环境的数据, 实际数据也分几档噪音随机
        noise_level = self.config['noise_level']
        #print('Random Noise Data added with noise level ' + str(noise_level))
        fls = []
        for fl in self.config['floor_list']:
            for pos in self.legal_pos[fl]:
                # 加入随机噪音
                if pos in self.poi[fl]:
                    noisy_data = int(np.random.rand() * noise_level - noise_level/2)
                    self.data_amount[fl][pos] += noisy_data
                    if self.data_amount[fl][pos] != self.data_amount_init[fl][pos]:
                        fls.append(1)
        assert 1 in fls
                    
    def _init_uav(self):
        #TODO: 初始化uav的各种参数，这个应该从config中读取，目前只是给定一个初始位置
        self.initial_pos = [self.config['uav_init_pos'] if i < int(self.config['agent_num']/2) else self.config['uav_init_pos_2'] for i in range(self.config['agent_num'])] #[(('site2','F3'), pos),(('site2','F3'),pos)]
        self.uav_pos = [self.config['uav_init_pos'] if i < int(self.config['agent_num']/2) else self.config['uav_init_pos_2'] for i in range(self.config['agent_num'])] # 记录所有的位置

    def _find_observations(self, agent):
        # 整合所有喂给智能体的observation信息，返回一个list
        # 这里仅对一个智能体找,即给出的observation 是agent的, 这里的agent是一个序号0,1,2,3，...
        # WARNING:注意这里的convention是 [每个位置数据量|...]这个涉及到坐标转化
        # [每个位置数据量|uav位置|]
        ret = []

        # 第一步：对于每个智能体，整理出当前观测范围内的所有的数据量，其余全部mask为0
        # TODO：CTDE时候喂所有人的观测
        
        # 整理当前agent观测内的数据量
        if self.config['use_global_observation_only']:
            # 这个直接给出哪里是POI,POI且有数据的地方为1，其余地方都是0
            temp_ret = []
            for fl in self.config['floor_list']:
                for pos in self.legal_pos[fl]:
                    if pos in self.poi[fl] and self.data_amount[fl][pos] > 0:
                        temp_ret.append(1)
                    else:
                        temp_ret.append(0)
            ret.extend(temp_ret)
        else:
            floor = self.uav_pos[agent][0]
            amount_in_sights = []
            for i in range(-self.config['observation_range'],self.config['observation_range']):
                for j in range(-self.config['observation_range'],self.config['observation_range']):
                    p = (self.uav_pos[agent][1][0] + i * self.config['poi_distance'], 
                            self.uav_pos[agent][1][1] + j * self.config['poi_distance']) #TODO: 3.15 这里观测有bug，没加全
                    if p in self.poi[floor]:
                        amount_in_sights.append(p)

            # 生成观测，观测是整个地图的legal pos的1-hot vector    
              
            # 6-27 TODO：观测中的 
            temp_ret = []
            for fl in self.config['floor_list']:
                if self.uav_pos[agent][0] != fl:
                    temp_ret.extend([0 for _ in range(len(self.poi[fl]))])
                else:
                    for p in self.poi[self.uav_pos[agent][0]]:
                        if p in amount_in_sights:
                            if self.malicious_flag[agent]:
                                temp_ret.append(self.data_amount[self.uav_pos[agent][0]][p])
                                #temp_ret.append(0)
                            else:
                                temp_ret.append(self.data_amount[self.uav_pos[agent][0]][p])
                        elif p in self.poi[fl] and self.data_amount[self.uav_pos[agent][0]][p] != 0:
                            temp_ret.append(1)
                        else:
                            temp_ret.append(0)
                    
            ret.extend(temp_ret)  #和legal position 同序 
        
        # 第二步：malicious是否干扰了：
        ret.extend([1 if self.malicious_flag[agent] else 0 for _ in range(5)])
    
        # 第三步：智能体的位置
        uav_pos_list = [] #[0,300,300,1,600,400] 格式为 层，具体位置
        for uav_p in self.uav_pos:
            uav_pos_list.append(self.config['floor_list'].index(uav_p[0])) # 层数
            uav_pos_list.extend(uav_p[1]) # 具体层中位置
        ret.extend(uav_pos_list)

        # 第四步：当前步数
        ret.extend([self.game_steps])


        return ret

    def _construct_transition_observation(self,uav_idx,action):
        # 喂给VIME探索的信息，和训练策略的信息有所不同，目的是用来预测转移概率
        # 喂给探索的states应该有这么几部分：第一包括state位置信息，当前状态数据量
        # 预测下一个state的位置，以及该处数据量
        # 位置用 onehot处理一下。
        curr_pos = []
        # print(uav_idx)
        # print(self.uav_pos[uav_idx][0])
        # print(self.config['floor_list'])
        curr_pos.append(self.config['floor_list'].index(self.uav_pos[uav_idx][0])) # 层数
        curr_pos.extend(self.uav_pos[uav_idx][1]) # 具体层中位置
    
        # # 动作，用onehot编码
        # action = [0 if i != action else 1 for i in range(self.config['total_action_dim'])]
        # assert self.config['total_action_dim'] < 32
        # action = [0,0,0,0,0,0] # 目前动作28
        # assert type(action) == int
        # binary_action = bin(action)[2:]
        # for i in range(len(binary_action)):
        #     action[len(action) - len(binary_action) + i] = binary_action[i]
        curr_pos.extend([100 if itm == 1 else 0 for itm in action])
        return np.array(curr_pos)


    def _vime_obs_process(self,obs,actions):
        ret = []
        #print(obs.shape,actions.shape)
        for agents in range(self.agent_num):
            ret.append(self._construct_transition_observation(agents,actions))
        #print(np.array(ret).shape)
        return np.array(ret)
    
    def _action_binary_int_transform():
        # 这里将动作onehot处理一下
        return
    
    def _position_onehot_transform(self,pos, usage = 'one_hot_to_floor'):
        # 这个函数用来将楼层位置和 one hot位置互相转化
        # pos 起始点要从0开始
        if usage == 'one_hot_to_floor':
            # 此时的pos是一个值
            assert type(pos) == int
            fl = self.config['floor_list'][0]
            ret_pos = self.legal_pos[fl][0]
            accum_idx = 0
            for floor in self.config['floor_list']:
                accum_idx += len(self.legal_pos[floor])
                if pos - accum_idx > 0:
                    # 楼层没到，下一层
                    continue
                else:
                    fl = floor
                    ret_pos = pos - accum_idx
                    break
            return(fl, self.legal_pos[fl][ret_pos])

        elif usage == 'floor_to_one_hot':
            # 此时的pos是一个list
            # [(('site2','F4'),(500,600)),...]
            assert type(pos) == list
            floor_length = [len(item) for item in list(self.legal_pos.values())] # 路网长度[100,210,...]
            vector_length = sum(floor_length) # 所有层路网的个数和
            ret = [0 for _ in range(vector_length)] 
            for p in pos:
                idx = self.legal_pos[p[0]].index(p[1])
                if self.config['floor_list'].index(p[0]) == 0:
                    ret[idx] = 1
                else:
                    ret[idx + sum(floor_length[:self.config['floor_list'].index(p[0]) - 1])] = 1
            return ret #返回one-hot list
        
    def _init_elevator_reward(self):
        # 第一步：对于所有的电梯处，以及每一层做一个分类，Kmeans就可以, 将POI都归类
        # 这个应该hardcode一下, load出来方便调用
        d = lambda x,y: np.sqrt((x[0]-y[0])**2 + (x[1]-y[1])**2)
        if os.path.exists('./intermediate/data/elevator_rewards.pkl'):
            with open('./intermediate/data/elevator_rewards.pkl', 'rb') as f:
                print('levator reward loaded, please check for correctness!')
                self.classification_result = pickle.load(f)
        else:
            self.classification_result = {} # {fl1:{elevator_0:[(100,200),...]}}
            for fl in self.config['floor_list']:
                self.classification_result[fl] = {}
                el_str = list(self.elevator[fl].keys())
                for i in range(len(el_str)): 
                    try :
                        assert self.elevator[fl][el_str[i]] in self.legal_pos[fl]
                    except AssertionError:
                        print(str(fl) + ' ' + el_str[i] + ' not in legal pos.')
                        exit(1)
                floor_classification_dict = {el_str[i]:[] for i in range(len(el_str))}

                for pos in self.poi[fl]: # 只对POI
                    min_idx = 0
                    min_d = d(pos, self.elevator[fl][el_str[0]])
                    for i in range(1, len(el_str)):
                        curr_d = d(pos,  self.elevator[fl][el_str[i]])
                        if curr_d < min_d:
                            min_idx = i
                            min_d = curr_d
                    floor_classification_dict[el_str[min_idx]].append(pos)

                self.classification_result[fl] = floor_classification_dict

            #6-29：记录上一个episode的每个点被谁采集的, 由于顺序采集 初始化给最后一个无人机，他不挤兑任何人
            # 每一次把last occupancy 更新为current 然后重置current
            self.last_occupancy = {fl:{pos:self.config['agent_num'] - 1 for pos in self.poi[fl]} for fl in self.config['floor_list']} #{fl1:{(100,200): uav_index}}
            self.current_occupancy = {fl:{pos:self.config['agent_num'] - 1 for pos in self.poi[fl]} for fl in self.config['floor_list']} #{fl1:{(100,200): uav_index}}

    
    def _determin_elevator_reward(self, agent):
        # 这是一个动态鼓励上下楼的算法
        # 返回格式：{('site2','F3'): {(elevator_pos: {'upstairs': -100, 'downstairs':100})}}
        # 6-26; # 返回格式：{('site2','F3'): {(elevator_pos: {'0': -100, '1':100, '2';100})}}
        # 如果遇到没法upstairs和downstairs的楼层情况，那keys就一个
 
        # 根据当前数据量算一个大概奖励，这个奖励谁先到谁先得 #TODO：这是devision of work得轨迹思路
        data_reward_dict = {}
        for fl in self.config['floor_list']:
            data_reward_dict[fl] = {}
            for elevator in self.elevator[fl].keys():
                data_reward_dict[fl][elevator] = sum([self.data_amount[fl][pos] \
                            if self.last_occupancy[fl][pos] >= agent else 0 for pos in self.classification_result[fl][elevator]])
                
         
        ret = {fl:{ele:{} for ele in self.elevator[fl].keys()} for fl in self.config['floor_list']}
        for fl_idx in range(len(self.config['floor_list'])):
            fl = self.config['floor_list'][fl_idx]
            for ele in self.elevator[fl].keys():
                for fls in range(len(self.config['floor_list'])):
                    ret[fl][ele][str(fls)] =  data_reward_dict[self.config['floor_list'][fls]][ele] \
                        - data_reward_dict[self.config['floor_list'][fl_idx]][ele]
        return ret
  
    def test(self,data):
        print(data)
        return [1]
    
    def _init_vime(self):
        self.transition_prob_model = VI.build_network(self.obs_dim, self.obs_dim)
        self.vime_pool = VI.SimpleReplayPool(500,(self.obs_dim,),self.config['total_action_dim'])
        self.current_observation_history = [] # 用来记录观测历史数据
        

    def _reset_vime(self):
        self.current_observation_history = []

    def _vime_update_rewards(self, obs, actions, dones, step):
        vime_reward = []
        for step in range(1, self.config['horizon']-1):
            temp_reward = []      
            for agents in range(self.config['agent_num']):    
                self.vime_pool.add_sample(obs[step][agents],[],[],[])
                data = (agents, self.transition_prob_model, torch.from_numpy(np.array(obs[step-1])), \
                        torch.from_numpy(np.array(obs[step])))    
                temp_reward.append(self._determine_vime_reward(data))
            vime_reward.append(temp_reward)
        return vime_reward
    
    def _determine_vime_reward(self, data):
        # 计算local的奖励，用KL来近似提高运算速度
        uav_idx, bnn_model, current_obs, next_obs = data
        VI.KL_preprocess(bnn_model, current_obs, next_obs)
        KL = VI.fisher_information_KL_approximate(VI.get_all_thetas(bnn_model))

        if len(self.trajectory_median[uav_idx]) == 0:
            reward = min(KL,1)
        else:
            reward = KL/median(self.trajectory_median[uav_idx])
        # 更新trajectory_mean
        self.trajectory_median[uav_idx].append(KL)
        if len(self.trajectory_median[uav_idx]) > self.config['median_list_length']:
            self.trajectory_median[uav_idx].pop(0)
        return reward
    
    def _terminate_condition(self):
        #到达终止时间后返回True
        if self.game_steps >= self.horizon:
            return True
        return False

    def _action_1_2_transform(self, in_actions, usage = '2_to_1'):
        # 由于用链接矩阵表示，因此动作用2维更方便，但是喂给神经网络需要压成一维，因此把对应关系在这里明确了
        # 注意坐标中心在矩阵中心位置, 图例如下，这么做是为了方便在图像里直接操作，坐标轴和图像坐标轴方向一致
        #  (-1,-1) (-1,0) (-1,1)         0 1 2
        #   (0,-1) (0,0) (0,1)           3 4 5   4不动即采数据
        #   (1,-1) (1,0) (1,1)           6 7 8
        #                                9：上楼梯 10：下楼梯
        # 因此 '1_to_2' 给一个5返回(0,1) '2_to_1'给一个(0,1)返回5

        # [,upstairs,downstaris,superconnect]

        length = self.config['neighbour_range'] * 2 + 1 # 矩阵边长，动作为方阵
        center_index = self.config['action_dim'] // 2  #(0,0)对应的index
        if usage == '1_to_2':
            if in_actions < self.config['action_dim']: #常规移动
                i = in_actions // length
                j = in_actions - (in_actions // length) * length
                return (i - self.config['neighbour_range'], j - self.config['neighbour_range']) # 整体坐标平移
            #多层楼
            elif in_actions < self.config['action_dim'] + self.config['floor_num']:
                num_fl = int(in_actions - self.config['action_dim'])
                return str(num_fl)
            # elif in_actions == self.config['action_dim']:
            #     # 上楼
            #     return 'upstairs'
            # elif in_actions == self.config['action_dim'] + 1:
            #     # 下楼
            #     return 'downstairs'
            elif in_actions == self.config['action_dim'] + self.config['floor_num']:
                # 超连接
                return 'superconnect'
            else :
                # 目前只有两个动作
                assert True == False  # 出现动作mapping错误
        elif usage == '2_to_1':
            list_length = self.config['total_action_dim']
            # 3.29 # TODO：大bug，好多沒對應上
            if in_actions == 'superconnect':
                return list_length - 1
            else:
                if type(in_actions) == str:
                    return list_length - self.config['floor_num'] -1 + int(in_actions)
                elif abs(in_actions[0]) <= self.config['neighbour_range'] and abs(in_actions[1]) <= self.config['neighbour_range']:
                    return in_actions[0] * length + in_actions[1] + center_index
                else:
                    assert True == False

    def _validate_action(self, action, pos):
        # 判断一个动作是否合法，包括是否撞墙，电梯是否能上
        # action: 17, pos:(('site2','f3'), (500,600))
        action_transformed = self._action_1_2_transform(action, '1_to_2')
        idx1 = self.legal_pos[pos[0]].index(pos[1]) # 由于adjacency matrix是按照legal_pos的index来建的图，recall _adjacency_matrix

        if action_transformed == 'superconnect':
            if pos[1] in self.super_lists_split[pos[0]]:
                # 当前位置在可以superconnect的位置
                return True
            else:
                return False
        elif type(action_transformed) == str:
            # 6-26: 上下楼不会出边界了

            if self.fl_map[pos[0]][idx1][idx1] == 2 and self.config['floor_list'].index(pos[0]) != int(action_transformed):
                return True
            else:
                return False
        else:
            # 给action后的位置找index
            actual_pos = (pos[1][0] + action_transformed[0] * self.config['poi_distance'], 
                pos[1][1] + action_transformed[1] * self.config['poi_distance'])
            try:
                idx2 = self.legal_pos[pos[0]].index(actual_pos)
            except ValueError as e:
                # 出了路网和边界
                return False
            if self.fl_map[pos[0]][idx1][idx2] != 0:
                # 只要不是0 证明在路网上
                return True
            return False

    def _update_pos(self, uav_index, transformed_action):
        # transformed_action 为2格式，即（-1,-1）样子的
        # 这里的action是必须已经过了validation的，确保没错
        d = lambda x,y: (x[0]-y[0])**2 + (x[1]-y[1])**2
        if transformed_action == 'superconnect':
            update_flag = False
            for tuple_pairs in self.super_lists[self.uav_pos[uav_index][0]]:
                t0 = tuple_pairs[0]
                t1 = tuple_pairs[1]
                if self.uav_pos[uav_index][1] == t0:
                    self.uav_pos[uav_index] = (self.uav_pos[uav_index][0],t1)
                    update_flag = True
                elif  self.uav_pos[uav_index][1] == t1:
                    self.uav_pos[uav_index] = (self.uav_pos[uav_index][0],t0)
                    update_flag = True
            assert update_flag == True
            
        elif type(transformed_action) == str:
            fl_index = self.config['floor_list'].index(self.uav_pos[uav_index][0])
            transformed_fl_index = int(transformed_action)
            # 找到电梯是哪一个, 通过电梯名字对应
            elevator_name = self.ele_pos_mapping[self.config['floor_list'][fl_index]][self.uav_pos[uav_index][1]]
            elevator_pos = self.elevator[self.config['floor_list'][transformed_fl_index]][elevator_name]

            # 上电梯后，楼层更新，位置到电梯口
            print('UAV ' + str(uav_index) + ' takes ' + elevator_name + ' on timestep '+ \
                  str(self.game_steps) + ' from ' + str(fl_index) + ' to ' + str(transformed_fl_index))      
            self.uav_pos[uav_index] = (self.config['floor_list'][transformed_fl_index], elevator_pos) 
        else:
            # 同层更新
            new_pos = (self.uav_pos[uav_index][1][0] + transformed_action[0] * self.config['poi_distance'],
               self.uav_pos[uav_index][1][1] + transformed_action[1] * self.config['poi_distance'])
            self.uav_pos[uav_index] = (self.uav_pos[uav_index][0], new_pos) 

    def _data_change_old(self, agent_num, transformed_action):
        maximum_collect = self.config['maximum_collect']
        # 3.5: 现在只要走到有reward就给，不需要必须是（0,0）
        if self.data_amount[self.uav_pos[agent_num][0]][self.uav_pos[agent_num][1]] >= maximum_collect:
            self.data_amount[self.uav_pos[agent_num][0]][self.uav_pos[agent_num][1]] -= maximum_collect
            return maximum_collect
        else:
            collected = self.data_amount[self.uav_pos[agent_num][0]][self.uav_pos[agent_num][1]]
            self.data_amount[self.uav_pos[agent_num][0]][self.uav_pos[agent_num][1]] = 0
            return collected
    
    def _data_change(self, agent_num, transformed_action):
        # 数据量变化
        # TODO: 数据变化量需要更改, 最大值应该写进env profile中去
        delta = {}
        maximum_collect = self.config['maximum_collect']
        collected = 0
        curr_fl = self.uav_pos[agent_num][0]
        curr_x = self.uav_pos[agent_num][1][0]
        curr_y = self.uav_pos[agent_num][1][1]
        visit_pos_list = []

        # 3.5: 现在只要走到有reward就给，不需要必须是（0,0）
        # 3.23：增强收集能力，现在移动路上都可以收集, manhatten distance
    
        if type(transformed_action) == str:
            # 上下楼的时候不用沿路收集了
            coll = min(maximum_collect, self.data_amount[self.uav_pos[agent_num][0]][self.uav_pos[agent_num][1]] )
            self.data_amount[self.uav_pos[agent_num][0]][self.uav_pos[agent_num][1]] = 0
            # AOI 更新
            delta[self.uav_pos[agent_num]] = self.game_steps - self.last_visit[curr_fl][(curr_x,curr_y)]
            self.last_visit[self.uav_pos[agent_num][0]][self.uav_pos[agent_num][1]] = self.game_steps
            self.aoi[self.uav_pos[agent_num][0]][self.uav_pos[agent_num][1]] = 0
            visit_pos_list.append(self.uav_pos[agent_num])
            return coll,delta,visit_pos_list
        
        for x in range(abs(transformed_action[0])):
            for y in range(x, abs(transformed_action[1])):                               
                if y < x:
                    # 沿路收集，不能来回绕
                    continue
                
                if transformed_action[0] >= 0:
                    pos_x = curr_x + x * self.config['poi_distance']
                else:
                    pos_x = curr_x - x * self.config['poi_distance']

                if transformed_action[1] >= 0:
                    pos_y = curr_y + y * self.config['poi_distance']
                else:
                    pos_y = curr_y - y * self.config['poi_distance']
                
                combine_pos = (pos_x, pos_y)
                if combine_pos not in self.legal_pos[curr_fl]:
                    continue

                left_collection_ability = maximum_collect - collected
                if self.data_amount[curr_fl][combine_pos] >= left_collection_ability:
                    self.data_amount[curr_fl][combine_pos] -= left_collection_ability
                    delta[(curr_fl,combine_pos)] = self.game_steps - self.last_visit[curr_fl][combine_pos]
                    self.last_visit[curr_fl][combine_pos] = self.game_steps
                    self.aoi[curr_fl][combine_pos] = 0
                    visit_pos_list.append((curr_fl,combine_pos))
                    return maximum_collect, delta, visit_pos_list
                else:
                    # 6-29: 这里记录了每个POI上一个episode被使用的情况
                    self.current_occupancy[curr_fl][combine_pos] = agent_num

                    collected += self.data_amount[curr_fl][combine_pos]
                    self.data_amount[curr_fl][combine_pos] = 0

                    delta[(curr_fl,combine_pos)] = self.game_steps - self.last_visit[curr_fl][combine_pos]
                    self.last_visit[curr_fl][combine_pos] = self.game_steps
                    self.aoi[curr_fl][combine_pos] = 0
                    visit_pos_list.append((curr_fl,combine_pos))

        return collected,delta,visit_pos_list     

        # 老版本，只有（0，0）收集    
        # if transformed_action == (0,0):
        #     if self.data_amount[self.uav_pos[agent_num][0]][self.uav_pos[agent_num][1]] >= maximum_collect:
        #         self.data_amount[self.uav_pos[agent_num][0]][self.uav_pos[agent_num][1]] -= maximum_collect
        #         return maximum_collect
        #     else:
        #         collected = self.data_amount[self.uav_pos[agent_num][0]][self.uav_pos[agent_num][1]]
        #         self.data_amount[self.uav_pos[agent_num][0]][self.uav_pos[agent_num][1]] = 0
        #         return collected
        # elif transformed_action == 'upstairs' or transformed_action == 'downstairs':
        #     #上楼或者下楼
        #     return 0
        # else:
        #     # 其他动作什么都不收集
        #     return 0

    def _masked_action(self):
        # 由于训练困难的事情，需要直接masked 掉所有的不可以被选中的动作
        # 在预处理阶段直接储存成一个dict {('site2'，'F3'):{(200,100):[0,0,1,1,..],...},...}
        # 其中 1代表动作可以选，0代表不可以
        # 返回load好的masked action dict
        # try:
        #     with open('./mask_actions.pickle', 'rb') as f:
        #         ret = pickle.load(f)
        #         print('masked_action loaded from saved directory \n')
        # except OSError:
        #     print('masked_action computed \n')
        ret = {fl:{pos:[] for pos in self.legal_pos[fl]} for fl in self.config['floor_list']}
        for fl in self.config['floor_list']:
            for pos in self.legal_pos[fl]:
                for i in range(self.action_dim):
                    # if i == self.action_dim - 2:
                    #     i == 'upstairs'
                    # elif i == self.action_dim - 1:
                    #     i == 'downstairs'
                    if self._validate_action(i, (fl,pos)):
                        ret[fl][pos].append(1)
                    else:
                        ret[fl][pos].append(0)
            # with open('./mask_actions.pickle', 'wb') as f:
            #     pickle.dump(ret,f)
            # print('masked_action saved \n')
        return ret

    def _init_aoi(self):
        # 作为环境的一部分，为每个poi记录aoi, 每次开始的时候要reset
        # 格式 {（’site‘，’Fl‘）：{(500,600): 10},...}
        self.aoi = {}
        self.last_visit = {}
        for fl in self.config['floor_list']:
            self.aoi[fl] = {}
            self.last_visit[fl] = {}
            for pos in self.legal_pos[fl]:
                self.aoi[fl][pos] = 0
                self.last_visit[fl][pos] = 0

    def _aoi_default_updates(self):
        # 基本update，由于时间步加1后所有位置aoi时间自动加1
        for fl in self.config['floor_list']:
            for pos in self.legal_pos[fl]:
                self.aoi[fl][pos] += 1

    def _aoi_updates_old(self, transformed_action, agent):
        # # 只有当action为采集的时候，才二次update aoi
        # if transformed_action == (2,2):
        #     fl = self.uav_pos[agent][0]
        #     pos = self.uav_pos[agent][1]
        #     delta = self.game_steps - self.aoi[fl][pos]
        #     self.aoi[fl][pos] = self.game_steps # 把访问时间设置为这一步时间
        #     return delta
        # else:
        #     # 其他动作均不算采集，只是路过
        #     return 0

        # 现在默认是走到就采集

        #3.31 TODO:有问题，没考虑过路采
        fl = self.uav_pos[agent][0]
        pos = self.uav_pos[agent][1]
        delta = self.game_steps - self.last_visit[fl][pos] # 上次访问时间 和这次访问时间的差
        self.last_visit[fl][pos] = self.game_steps 
        self.aoi[fl][pos] = 0 #访问后归0
        return delta

    def _aoi_updates(self, transformed_action, agent):
        pass

    def _compute_aoi_violation(self):
        #TODO: 如果变化数据量了，这里需要稍微修改
        # count_viloation_dict = {fl:{pos:1 if self.data_amount[fl][pos] == self.data_amount_init else 0 \
        #                             for pos in self.poi[fl]} for fl in self.config['floor_list']}
        num_poi = 0
        total = 0
        if 'use_changing_data':
            # 数据量在变
            # 统计按时被cover的比例
            if self.game_steps % self.config['data_last_time'] == self.config['data_last_time'] - 1:
                # 更新aoi_violated_ratio
                for fl in self.config['floor_list']:
                    for pos in self.poi[fl]:
                        num_poi += 1
                        if self.data_amount[fl][pos] == self.data_amount_init[fl][pos]:
                            total += 1      
                self.aoi_violated_ratio = total/(num_poi)

        return self.aoi_violated_ratio               
                
    def _reward_compute_old(self, agent, transformed_action ,delta_data_amount):
        # 在这里计算奖励
        if self.config['rewarding_methods'] == 'default':
            # 默认的奖励计算方式
            # 对于无人机采集的数据量，每一份给1

            # 默认每一步惩罚
            default_penalty = -2

            # 第一次上电梯给一个很大的奖励
            extra_reward = 0
            if transformed_action == 'upstairs' or transformed_action == 'downstairs': 
                if self.first_step[agent] > self.agent_num/2: 
                    # 只有第一次上楼探索才奖励
                    if transformed_action == 'upstairs':
                        extra_reward = 400
                    print('floor changed with extra rewards ',agent, 'at time step ', self.game_steps)
                    # 这里上完就全部改了，在reset的时候会被重置
                    for agentnum in range(self.agent_num):
                        self.first_step[agentnum] -=1
                else:
                    #print('floor changed with penalty ',agent, 'at time step ', self.game_steps)
                    extra_reward = -500
          
            different_floor_reward = 0
            # 无人机之间距离的鼓励，隔了楼层应该更大
            # for uavs in range(self.agent_num):
            #     if self.uav_pos[uavs][0] != self.uav_pos[agent][0]:
            #         # 和无人机agent不在一层
            #         different_floor_reward = 3
            #         break
            #     else:
            #         # 目前在同一层不奖励
            #         different_floor_reward = 0
            
            # 距离鼓励
            distance_reward = 0
            # distance_reward = self._distance_compute(agent)

            #print(distance_reward)
            return default_penalty + delta_data_amount + extra_reward + different_floor_reward + distance_reward

        # elif self.config['rewarding_methods'] == 'count_base':
        #     # 这里边加入了最简单的访问奖励，去鼓励访问更多的states
        #     pass

    def _reward_compute(self, agent, transformed_action ,delta_data_amount, 
                        bnn_model = None, current_obs = None, next_obs = None):
        # 在这里计算奖励
        if self.config['rewarding_methods'] == 'default':
            # 默认的奖励计算方式s
            # 对于无人机采集的数据量，每一份给1
            
            # 默认每一步惩罚
            default_penalty = -2
            ratio = 0
            ele_reward = 0
            #return default_penalty + delta_data_amount

            # # VIME奖励
            # vime_reward = 0
            
            # if bnn_model is not None:   
            #     vime_reward = self._determine_vime_reward(agent, bnn_model, current_obs, next_obs)
    
            if self.config['use_elevator']:

                if self.game_steps % self.config['data_last_time'] == 0:
                    # 7-4:数据刷新了，需要重新set lock
                    self.elevator_lock = np.zeros((len(self.elevator[list(self.elevator.keys())[0]]), int(self.config['data_changes_num']),\
                            int(self.config['floor_num'])))

                # 上下电梯奖励 TODO：有转移概率了用DP计算
                if type(transformed_action) == str and transformed_action != 'superconnect':
                    fl = self.uav_pos[agent][0]
                    pos = self.uav_pos[agent][1]
                    fl_index = self.config['floor_list'].index(self.uav_pos[agent][0])
                    transformed_fl_index = int(transformed_action)
                    # 找到电梯是哪一个, 通过电梯名字对应
                    elevator_name = self.ele_pos_mapping[self.config['floor_list'][fl_index]][self.uav_pos[agent][1]]
                    elevator_idx = list(self.elevator[self.config['floor_list'][transformed_fl_index]]).index(elevator_name)
                    
                    slot = int(min(self.game_steps // self.config['data_last_time'], self.config['data_changes_num']-1))
                    
                
                    if self.elevator_lock[elevator_idx][slot][transformed_fl_index] == 0:
                        # 1. 电梯根据乘坐电梯到达某一层来给
                        # 2. 一个时间段，奖励只给第一个用这个电梯的人，用同一个电梯到同一层要惩罚
                        if self.elevator_usage_time[agent] == 0 or abs(self.elevator_usage_time[agent] - self.game_steps) > self.config['data_last_time']/2 :
                            ele_reward = self._determin_elevator_reward(agent)[fl][self.ele_pos_mapping[fl][pos]][transformed_action]
                            self.elevator_lock[elevator_idx][slot][transformed_fl_index] = 1
                        else:
                            ele_reward = -40
                    else:
                        # 不可以使用电梯正奖励并且考虑适当惩罚
                        ele_reward = -40
                        #ele_reward = min(self._determin_elevator_reward()[fl][self.ele_pos_mapping[fl][pos]][transformed_action], 0)

                    if self.game_steps > self.config['horizon'] - 20:
                        # 防止偷奖励的
                        ele_reward = 0

                    self.elevator_usage_time[agent] = self.game_steps
                        
            return default_penalty + delta_data_amount + ele_reward * ratio

        # elif self.config['rewarding_methods'] == 'count_base':
        #     # 这里边加入了最简单的访问奖励，去鼓励访问更多的states
        #     pass

    def _elevator_DP_reward(self,obs):
        # feed in current obs, compute_DP_reward
        pass
    
    def _aoilambda(self,dummy):
        # import copy
        # cpy = copy.deepcopy(self.aoi_lambda)
        print(self.aoi_lambda[0][('site2', 'F2')][(580, 230)])
        print(self.aoi_lambda[0][('site2', 'F3')][(580, 230)])
        print(self.aoi_lambda[0][('site2', 'F4')][(580, 230)])
        return self.aoi_lambda
    
    def _init_lambda_aoi(self):
        # 对偶梯度拉格朗日乘子初始化
        # 每一个poi都有一个乘子对应， 目前初始化方式是[0,1]间随机
        # 这里对应的是legal_pos的dictionary，对于所有不在poi dict里的，lambda为0不变

        self.aoi_lambda = {i:{fl:{pos:0 if pos not in self.poi[fl] else np.random.rand() \
            for pos in self.legal_pos[fl]} for fl in self.config['floor_list']} for i in range(self.config['data_changes_num'])}
        
    def _lambda_projection(self,lambda_list, default_upperbound = float('inf')):
        # 由于拉格朗日乘子是大于等于0的，因此需要投影进行投影
        # lambda_list :[lambda_1, lambda_2,,...] 顺序和self.poi中的顺序保持一致
        # 控制投影上限的，例如设成1将会投影到[0,1]^P 这个l1-ball上

        # 这是一个带约束的least square的问题
        # TODO：这是一个convex programming，slater也明显成立，用分类讨论KKT加速求解

        A = np.identity(len(lambda_list)) 
        b = np.array(lambda_list)
        s = np.array([default_upperbound for _ in range(len(lambda_list))])
        x = cp.Variable(len(lambda_list))
        prob = cp.Problem(cp.Minimize(cp.sum_squares(A@x - b)),[default_upperbound >= x, x >= 0])
        prob.solve()
        # # Print result.
        # print("\nThe optimal value is", prob.value)
        # print("A solution x is")
        # print(np.round(x.value, 3))
        # print("A dual solution is")
        # print(prob.constraints[0].dual_value)
        return np.round(x.value,3)

    def _lambda_projection_KKT(self,lambda_list,default_upperbound = float('inf')):
        # 果然cvxpy会求崩掉
        # 所以用KKT implies 最优分类讨论
        ret = []
        for lambdas in lambda_list:
            if lambdas < 0:
                ret.append(0)
            elif lambdas > default_upperbound:
                ret.append(default_upperbound)
            else:
                ret.append(lambdas)
        return ret

    def _constraints_aoi_reward(self, agent_rewards, agent_num, visit_pos_list):
        # 条件AOI
        # 目前奖励给的方式是先转移，然后采集转移到位置的数据, 这里也要保持一致
        
        # TODO： 目前为全部收集完，所以AOI归0
        # 4-1 :现在给奖励也要沿路给了

        ret_rewards = agent_rewards

        # 这里需要做一次clip一下AOI 防止两次之间时间过长但是被认为是好的的情况
        for pos in visit_pos_list:
            ret_rewards += self.aoi_lambda[pos[0]][pos[1]] * min(self.aoi[pos[0]][pos[1]], self.config['dual_clip_value'])

        return ret_rewards

    def _compute_lambda_updates(self, constraint_dict):
        # 这里来计算lambda的更新
        # constraint_dict :{fl:{pos: sum_of_constraint_reward}}

        # constraint_dict = n_{m,p} - c_{m,p}
        for i in range(self.config['data_changes_num']):
            for fl in self.config['floor_list']:
                for pos in self.poi[fl]:
                    self.aoi_lambda[i][fl][pos] -= self.config['dual_descent_learning_rate'] * constraint_dict[i][fl][pos]
        
        #投影到拉格朗日限定区间
        lambda_list = [self.aoi_lambda[i][fl][pos] for i in range(self.config['data_changes_num']) \
                       for fl in self.config['floor_list'] for pos in self.legal_pos[fl]]
        lambda_list = self._lambda_projection_KKT(lambda_list,self.config['dual_upperbound'])

        count = 0
        for i in range(self.config['data_changes_num']):
            for fl in self.config['floor_list']:
                for pos in self.legal_pos[fl]:
                    self.aoi_lambda[i][fl][pos] = lambda_list[count]
                    count += 1
                
    def _distance_compute(self, agent):
        # 为agent计算和其他所有agent的距离和
        # 不在同一层按照 一个固定距离 + 坐标d计算

        # 差楼层固定距离 200
        d_default_2 = 500
        # 控制总量的系数
        rate = 0.003
        total_d_2 = 0

        for i in range(self.agent_num):
            if self.uav_pos[i][0] == self.uav_pos[agent][0]:
                # 在同一层
                total_d_2 += (self.uav_pos[i][1][0] - self.uav_pos[agent][1][0])**2 \
                    + (self.uav_pos[i][1][1] - self.uav_pos[agent][1][1])**2
            else:
                # 不在同一层
                total_d_2 +=  d_default_2
        return np.sqrt(total_d_2) * rate
    
    def _compute_total_data(self, mode = 'init', period = None):
        # 计算场景里一开始的数据量的和
        # 用于看收集率的, init计算初始数据量的，runtime 看当前剩余的数据量
        # period 是用来给change data看这是第几次变化的，[0, num_changes]
        total = 0
        if mode == 'init':
            for fl in self.config['floor_list']:
                for pos in self.legal_pos[fl]:
                    total += self.data_amount_init[fl][pos]
        elif mode == 'init_with_changes':
            #TODO: 当数据变化的时候要注意修改 data_amount_init和这里
            for fl in self.config['floor_list']:
                for pos in self.legal_pos[fl]:
                    total += self.data_amount_init[fl][pos]
        elif mode == 'runtime':
            for fl in self.config['floor_list']:
                for pos in self.legal_pos[fl]:
                    total += self.data_amount[fl][pos]
        return total

    def _init_data_amount(self):
        # 初始化数据量
        if self.config['use_changing_data']:
            data = 0
            for i in range(self.config['data_changes_num']):
                data += self._compute_total_data('init_with_changes', period=i) 
        else:
            data = self._compute_total_data('init')
        return data
        
    

if __name__ == '__main__':
    env = EnvCore()
    # pos = env._load_red_from_maps(f'./output/mod_floorplan/site2/F4_POI_on_grid.png') 
    # print(len(pos))
    # pos = env._load_blue_from_maps(f'./output/mod_floorplan/site2/F4_POI_on_grid.png')
    # print(len(pos))

    # adj = env._adjacency_matrix(pos, 1035, 900, 10) #第三项后续应该放在config里

    print(env._lambda_projection([0.1, -0.2, -1], 0.05))
    exit(1)

    with open('./mask_actions.pickle', 'rb') as f:
        test = pickle.load(f)

    for i in range(200):
        pos1 = env.uav_pos[0]
        pos2 = env.uav_pos[1]
        idx1 = list(set([i if test[pos1[0]][pos1[1]][i] == 1 else -1 for i in range(env.action_dim)]))
        idx2 = list(set([i if test[pos2[0]][pos2[1]][i] == 1 else -1 for i in range(env.action_dim)]))
        idx1.remove(-1)
        idx2.remove(-1)
        action1 = np.random.choice(idx1)
        action2 = np.random.choice(idx2)
        if action1 == 25 or action1 == 26 or action2 == 25 or action2 == 26:
            print(action1, idx1,'\n')
        env.step([[1 if i == action1  else 0 for i in range(env.action_dim)],
        [1 if i == action2  else 0 for i in range(env.action_dim)]])
