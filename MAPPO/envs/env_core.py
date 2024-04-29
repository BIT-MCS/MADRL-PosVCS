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
cwd = os.getcwd()
sys.path.append(os.path.join(cwd, 'MAPPO'))
sys.path.append(os.path.join(cwd, 'prepare_data'))
sys.path.append(os.path.join(cwd, 'MI_intrinsic'))
from obstacle import _choose_color
from compute_f import split_ts_seq, compute_step_positions
from io_f import read_data_file
from visualize_f import visualize_trajectory, visualize_heatmap, visualize_floor_plan, save_figure_to_html, save_figure_to_png
from obstacle import *
from obstacle import _visualize_poi_on_map
from data_process import dict_of_grid_data, data_prediction,load_GP, post_process_and_save_GP,connect_all_parts,connected_component
from MAPPO.env_config import F2_3_floors 
import site2_handcraft as handcraft
import cvxpy as cp
from skimage.draw import line
import VI
from statistics import median
import torch


class EnvCore(object):
    def __init__(self):
        self.config = F2_3_floors.config 
        if self.config['if_read_states_besides_config']:
            flag = True
        
        if self.config['if_read_states']:
            with open('./intermediate/data/states_info.pkl', 'rb') as f:
                myinstance = pickle.load(f)

            for k in myinstance.__dict__.keys():
                setattr(self, k, getattr(myinstance, k))
            if flag:
                self.config = F2_3_floors.config 
            print('States Info Loaded.')
            self.total_data = self._init_data_amount() 
            self.runtime_data = 0
            self.aoi_violated_ratio = 0 
        else:
            self.agent_num = self.config['agent_num'] 
            self.malicious_flag = [False for _ in range(self.agent_num)] 
            self.action_dim = self.config['total_action_dim'] 
            self.game_steps = 0
            self.horizon = self.config['horizon']
            self._state_init()
            self._init_aoi()
            self._init_elevator_reward()
            

            self.total_data = self._init_data_amount() 
            self.runtime_data = 0
            self.aoi_violated_ratio = 0 
        
            if self.config['use_dual_descent']:
                self._init_lambda_aoi()

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
        sub_agent_obs = []
        self.uav_pos = [self.config['uav_init_pos'] if i < int(self.config['agent_num']/2) else self.config['uav_init_pos_2'] for i in range(self.config['agent_num'])] 
        self.game_steps = 0
        self.data_amount = deepcopy(self.data_amount_init)
        self._data_amount_true_env()
        self._init_aoi() 
        self.first_step = [self.agent_num for _ in range(self.agent_num)] 
        self.elevator_usage_time = [0 for _ in range(self.agent_num)] 
        self.elevator_lock = np.zeros((len(self.elevator[list(self.elevator.keys())[0]]), int(self.config['data_changes_num']),\
                int(self.config['floor_num']))) 
        self.trajectory_median = [[] for _ in range(self.agent_num)] 

        self.last_occupancy = deepcopy(self.current_occupancy)

        self.current_occupancy = {fl:{pos:self.config['agent_num'] - 1 for pos in self.poi[fl]} for fl in self.config['floor_list']} 

        for i in range(self.agent_num):
            sub_obs = self._find_observations(i)
            sub_agent_obs.append(sub_obs)
        return sub_agent_obs

    def step(self, actions, rend = False):
        sub_agent_obs = []
        sub_agent_reward = []
        sub_agent_done = []
        sub_agent_info = []
        self.game_steps += 1

        self._aoi_default_updates()

        if self.config['use_changing_data']:
            if self.config['use_same_data'] and self.game_steps % self.config['data_last_time'] == 0:
                self.data_amount = deepcopy(self.data_amount_init)                

        for i in range(self.agent_num):

            action_selected = list(actions[i]).index(1)
            transformed_action = self._action_1_2_transform(action_selected, usage = '1_to_2')
    
            delta_data_amount, aoi_delta, visit_pos_list = self._data_change(i,transformed_action) 
            self.runtime_data +=  delta_data_amount
            info_dict = {}
            if rend:
                self.take_out_dict = {}

            if self._validate_action(action_selected, self.uav_pos[i]):
                if rend and self.config['use_malicious_evaluate']:
                    if self.data_amount[self.uav_pos[i][0]][self.uav_pos[i][1]] > 0:
                        transformed_action = (0,0)
                        self.malicious_flag[i] = True
                        r = 1 
                    else:
                        p_tran = self.transition_prob[self.uav_pos[i][0]][self.uav_pos[i][1]]
                        tuple_list = [transformed_action, (0,0)]
                        idx = np.random.choice([0,1], 1, \
                                        p = [p_tran, 1 - p_tran])[0]
                        true_action = tuple_list[idx]
                        r = 0

                elif self.config['malicious_exist']:
                    if  self.malicious_flag[i] == False and delta_data_amount > 0:
                        transformed_action = (0,0)
                        self.malicious_flag[i] = True
                        r = self._reward_compute(i,transformed_action,delta_data_amount)    
                    else:
                        self.malicious_flag[i] = False
                        r = -10
                       
                else:
                    p_tran = self.transition_prob[self.uav_pos[i][0]][self.uav_pos[i][1]]
                    tuple_list = [transformed_action, (0,0)]
                    idx = np.random.choice([0,1], 1, \
                                    p = [p_tran, 1 - p_tran])[0]
                    true_action = tuple_list[idx]
                    r = self._reward_compute(i,true_action,delta_data_amount)


                sub_agent_reward.append([r])

             
                self._update_pos(i,transformed_action)
            else: 
                print('failed action transformed ',self.game_steps)
                sub_agent_reward.append([0])

            if self.config['use_masked_action']:
                info_dict['available_actions'] = self.masked_action[self.uav_pos[i][0]][self.uav_pos[i][1]]

            if self.config['use_dual_descent']:
                info_dict['aoi_reduce'] = aoi_delta

            sub_agent_obs.append(self._find_observations(i))
        

            if self._terminate_condition(): 
                sub_agent_done.append(True) 
            else:
                sub_agent_done.append(False)
            
            if rend:
                keys = ['current_position' ,'transformed_action','pos_info','available_actions', 'collection_ratio', 'violation_ratio']
                pos_dict = {'pos':{fl:[pos for pos in self.legal_pos[fl]] for fl in self.config['floor_list']},
                        'val':{fl:[self.data_amount[fl][pos] for pos in self.legal_pos[fl]] for fl in self.config['floor_list']}}
                info_dict = {keys[0]: self.uav_pos[i], keys[1]:transformed_action, keys[2]:pos_dict}
                info_dict[keys[3]] = self.masked_action[self.uav_pos[i][0]][self.uav_pos[i][1]]
                info_dict[keys[4]] = self.runtime_data / self.total_data
                info_dict[keys[5]] = self._compute_aoi_violation()

            sub_agent_info.append(info_dict)
            

            self.visit_counts[self.uav_pos[i][0]][self.uav_pos[i][1]][i] += 1
        if rend:
            self.take_out_dict = deepcopy(info_dict)

        return [sub_agent_obs, sub_agent_reward, sub_agent_done, sub_agent_info]


    
    def _runtime_summary(self):

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
        self._init_picture() 
        self._init_poi() 
        self._adjacency_matrix() 
        self._data_amount_extract(dict_of_grid_data(self.config['floor_list'], self.legal_pos)) 
        self._init_transition() 
        self.elevator_transition_prob = {}
        self.masked_action = self._masked_action() 
   
        self._init_uav()
        self.initial_obs = [self._find_observations(i) for i in range(self.config['agent_num'])]
        self.obs_dim = len(self.initial_obs[0])

        self._init_aoi() 
        self.first_step = [self.agent_num for _ in range(self.agent_num)] 
        self.elevator_usage_time = [0 for _ in range(self.agent_num)]
        self.elevator_lock = np.zeros((len(self.elevator[list(self.elevator.keys())[0]]), int(self.config['data_changes_num']),\
                                      int(self.config['floor_num'])))
 
        self.trajectory_median = [[] for _ in range(self.agent_num)]
    
    def _init_picture(self):
        
        self.picture_dict = {k:[] for k in self.config['floor_list']}
        self.black_array = {k:[] for k in self.config['floor_list']}
        for keys in self.picture_dict:
            site = keys[0]
            floor = keys[1]
            
            maplist = [
                
                f'./prepare_data/output/mod_floorplan/{site}/{floor}_POI_on_grid.png' 
            ]
            
            im = cv.imread(f'./prepare_data/output/mod_floorplan/{site}/{floor}_env.png')
            self.black_array[keys] = im
            self.picture_dict[keys].extend(maplist)
        self.x = {k:cv.imread(self.picture_dict[k][0]).shape[0] for k in self.picture_dict.keys()} 
        self.y = {k:cv.imread(self.picture_dict[k][0]).shape[1] for k in self.picture_dict.keys()} 
    
    def _init_transition(self):
        
        
        self.transition_prob = {fl:{pos:0 for pos in self.legal_pos[fl]} for fl in self.config['floor_list']}
        
        if self.config['transition_mode'] == 'full_success':
            
            for fl in self.config['floor_list']:
                for pos in self.legal_pos[fl]:
                    self.transition_prob[fl][pos] = 1
        elif self.config['transition_mode'] == 'with_random_collision':
            np.random.seed(100)
            
            problist = 1 - 0.3 * np.random.rand(sum([len(self.legal_pos[fl]) for fl in self.config['floor_list']]))
            count = 0
            for fl in self.config['floor_list']:
                for pos in self.legal_pos[fl]:
                    self.transition_prob[fl][pos] = problist[count]
                    count += 1
        elif self.config['transition_mode'] == 'with_deterministic_collision':
            pass
        elif self.config['transition_mode'] == 'with_malicious_and_deterministic_collision':
            
            pass
            
    def _init_poi(self):
        
        self.poi = {k: self._load_red_from_maps(v[0]) for k,v in self.picture_dict.items()}
        self.legal_pos = {k: self._load_blue_from_maps(v[0]) + self._load_red_from_maps(v[0]) for k,v in self.picture_dict.items()}
        l = sum([len(self.poi[k]) for k in list(self.poi.keys())])
        print('Total ' + str(l) +' poi')

    def _load_red_from_maps(self, in_dir):
        
        img = cv.imread(in_dir)
        red = np.array([0,0,255])
        pos = _choose_color(in_dir, red)
        return pos

    def _load_blue_from_maps(self, in_dir):
        
        img = cv.imread(in_dir)
        blue = np.array([255,0,0])
        pos = _choose_color(in_dir, blue)
        return pos
    
    def _adjacency_matrix(self):

        
        self.elevator = {k:getattr(handcraft, f'{k[0]}_{k[1]}_elevator')() for k in self.config['floor_list']}
        
        
        def local_adjacency(pos, max_x, max_y, distance, fl):
            
            
            
            adj = np.zeros([len(pos), len(pos)])
            
            index_matrix = np.zeros([max_x, max_y]) 
            for p in pos:
                index_matrix[p[0]][p[1]] = 1

            
            for x in range(max_x):
                for y in range(max_y):
                    
                    if index_matrix[x][y] != 1:
                        continue
                    
                    idx1 = pos.index((x,y))
                    for i in range(self.config['neighbour_range'] * 2 + 1): 
                        for j in range(self.config['neighbour_range'] * 2 + 1): 
                                               
                            x1 = x + (i-self.config['neighbour_range']) * distance
                            y1 = y + (j-self.config['neighbour_range']) * distance
                            if x1 in range(max_x) and y1 in range(max_y) and index_matrix[x1][y1] == 1:
                                
                                if True:    
                                    idx2 = pos.index((x1, y1))
                                    adj[idx1][idx2] = 1
                                    
            assert (adj.T == adj).all()

            return adj 
          
        self.fl_map = {}
        for keys in self.config['floor_list']:
            self.fl_map[keys] = local_adjacency(self.legal_pos[keys],self.x[keys],self.y[keys], self.config['poi_distance'], keys) 
    
        self.ele_pos_mapping = {fl:{} for fl in self.fl_map.keys()} 
    
        for fl in self.fl_map.keys():
            for el in self.elevator[fl].values():
                ele_index = self.legal_pos[fl].index((el[0], el[1]))
                self.fl_map[fl][ele_index][ele_index] = 2
                
                for i in range(-self.config['up_down_range'],self.config['up_down_range']):
                    for j in range(-self.config['up_down_range'],self.config['up_down_range']):
                        pos = (el[0] + i * self.config['poi_distance'], 
                          el[1] + j * self.config['poi_distance'])
                        if pos in self.legal_pos[fl]:    
                            self.ele_pos_mapping[fl][pos] = list(self.elevator[fl].keys())[list(self.elevator[fl].values()).index(el)]
                            idx = self.legal_pos[fl].index(pos)
                            self.fl_map[fl][idx][idx] = 2
            
            assert (self.fl_map[fl].T == self.fl_map[fl]).all()

        
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
            
            assert (ret_list[fl_idx].T == ret_list[fl_idx]).all()
            new_fl_dict[self.config['floor_list'][fl_idx]] = ret_list[fl_idx]
        self.fl_map = new_fl_dict
          
    def _check_hiting_walls(self, pos1, pos2, fl):

        discrete_line = list(zip(*line(*pos1, *pos2)))
        for points in discrete_line:
            if (self.black_array[fl][points[0]][points[1]] == np.array([0,0,0])).all():
                return True
        return False
    
    def _data_amount_extract(self, data_dict, save_dict = False):
        
        
        self.data_amount = {k:{kp:0 for kp in self.legal_pos[k]} for k in data_dict.keys()}
        self.data_amount_init = {k:{kp:0 for kp in self.legal_pos[k]} for k in data_dict.keys()} 
        self.all_poi_num = sum([len(self.legal_pos[fs]) for fs in data_dict.keys()])
        total_amount = 0
        for fl in self.config['floor_list']:
            try :
                print(f'Load GP model at {fl[0]} {fl[1]} for data amount prediction\n')
                RSSI_dict, predict_dict = load_GP(fl[0], fl[1])
            except FileNotFoundError:
                print('Generating New GP at {fl[0]} {fl[1]} with default settings\n')
                save_dict = True 
                RSSI_dict, predict_dict = data_prediction(data_dict[fl], self.legal_pos[fl]) 
            if save_dict:
                print('A new model at {fl[0]} {fl[1]} is saved to current directory\n')
                post_process_and_save_GP(RSSI_dict, predict_dict, fl[0], fl[1]) 
          
            for ks in predict_dict.keys():
                for i in range(len(self.poi[fl])):
                    
                    pos = self.poi[fl][i]
                    idx = self.legal_pos[fl].index(pos)
                    
                    if predict_dict[ks][idx] == 1:
                        total_amount +=1
                        self.data_amount[fl][pos] += 1
                        self.data_amount_init[fl][pos] += 1
        print('Total amount of data: ' + str(total_amount))
     
    
    def _data_amount_true_env(self):
        
        noise_level = self.config['noise_level']
        
        fls = []
        for fl in self.config['floor_list']:
            for pos in self.legal_pos[fl]:
                
                if pos in self.poi[fl]:
                    noisy_data = int(np.random.rand() * noise_level - noise_level/2)
                    self.data_amount[fl][pos] += noisy_data
                    if self.data_amount[fl][pos] != self.data_amount_init[fl][pos]:
                        fls.append(1)
        assert 1 in fls
                    
    def _init_uav(self):
        
        self.initial_pos = [self.config['uav_init_pos'] if i < int(self.config['agent_num']/2) else self.config['uav_init_pos_2'] for i in range(self.config['agent_num'])] 
        self.uav_pos = [self.config['uav_init_pos'] if i < int(self.config['agent_num']/2) else self.config['uav_init_pos_2'] for i in range(self.config['agent_num'])] 

    def _find_observations(self, agent): 
        ret = []       
        if self.config['use_global_observation_only']:
            
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
                            self.uav_pos[agent][1][1] + j * self.config['poi_distance']) 
                    if p in self.poi[floor]:
                        amount_in_sights.append(p)
      
            temp_ret = []
            for fl in self.config['floor_list']:
                if self.uav_pos[agent][0] != fl:
                    temp_ret.extend([0 for _ in range(len(self.poi[fl]))])
                else:
                    for p in self.poi[self.uav_pos[agent][0]]:
                        if p in amount_in_sights:
                            if self.malicious_flag[agent]:
                                temp_ret.append(self.data_amount[self.uav_pos[agent][0]][p])
                                
                            else:
                                temp_ret.append(self.data_amount[self.uav_pos[agent][0]][p])
                        elif p in self.poi[fl] and self.data_amount[self.uav_pos[agent][0]][p] != 0:
                            temp_ret.append(1)
                        else:
                            temp_ret.append(0)
                    
            ret.extend(temp_ret)  
        
        
        ret.extend([1 if self.malicious_flag[agent] else 0 for _ in range(5)])
    
        
        uav_pos_list = [] 
        for uav_p in self.uav_pos:
            uav_pos_list.append(self.config['floor_list'].index(uav_p[0])) 
            uav_pos_list.extend(uav_p[1]) 
        ret.extend(uav_pos_list)

        
        ret.extend([self.game_steps])


        return ret

    def _construct_transition_observation(self,uav_idx,action):
        
        curr_pos = []
        curr_pos.append(self.config['floor_list'].index(self.uav_pos[uav_idx][0])) 
        curr_pos.extend(self.uav_pos[uav_idx][1]) 
        curr_pos.extend([100 if itm == 1 else 0 for itm in action])
        return np.array(curr_pos)


    def _vime_obs_process(self,obs,actions):
        ret = []
        
        for agents in range(self.agent_num):
            ret.append(self._construct_transition_observation(agents,actions))
        
        return np.array(ret)
    
    def _action_binary_int_transform():
        
        return
    
    def _position_onehot_transform(self,pos, usage = 'one_hot_to_floor'):
        
        
        if usage == 'one_hot_to_floor':
            
            assert type(pos) == int
            fl = self.config['floor_list'][0]
            ret_pos = self.legal_pos[fl][0]
            accum_idx = 0
            for floor in self.config['floor_list']:
                accum_idx += len(self.legal_pos[floor])
                if pos - accum_idx > 0:
                    
                    continue
                else:
                    fl = floor
                    ret_pos = pos - accum_idx
                    break
            return(fl, self.legal_pos[fl][ret_pos])

        elif usage == 'floor_to_one_hot':
            
            
            assert type(pos) == list
            floor_length = [len(item) for item in list(self.legal_pos.values())] 
            vector_length = sum(floor_length) 
            ret = [0 for _ in range(vector_length)] 
            for p in pos:
                idx = self.legal_pos[p[0]].index(p[1])
                if self.config['floor_list'].index(p[0]) == 0:
                    ret[idx] = 1
                else:
                    ret[idx + sum(floor_length[:self.config['floor_list'].index(p[0]) - 1])] = 1
            return ret 
        
    def _init_elevator_reward(self):
        
        
        d = lambda x,y: np.sqrt((x[0]-y[0])**2 + (x[1]-y[1])**2)
        if os.path.exists('./intermediate/data/elevator_rewards.pkl'):
            with open('./intermediate/data/elevator_rewards.pkl', 'rb') as f:
                print('levator reward loaded, please check for correctness!')
                self.classification_result = pickle.load(f)
        else:
            self.classification_result = {} 
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

                for pos in self.poi[fl]: 
                    min_idx = 0
                    min_d = d(pos, self.elevator[fl][el_str[0]])
                    for i in range(1, len(el_str)):
                        curr_d = d(pos,  self.elevator[fl][el_str[i]])
                        if curr_d < min_d:
                            min_idx = i
                            min_d = curr_d
                    floor_classification_dict[el_str[min_idx]].append(pos)

                self.classification_result[fl] = floor_classification_dict

            
            
            self.last_occupancy = {fl:{pos:self.config['agent_num'] - 1 for pos in self.poi[fl]} for fl in self.config['floor_list']} 
            self.current_occupancy = {fl:{pos:self.config['agent_num'] - 1 for pos in self.poi[fl]} for fl in self.config['floor_list']} 

    
    def _determin_elevator_reward(self, agent):
               
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
        self.current_observation_history = [] 
        
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
        
        uav_idx, bnn_model, current_obs, next_obs = data
        VI.KL_preprocess(bnn_model, current_obs, next_obs)
        KL = VI.fisher_information_KL_approximate(VI.get_all_thetas(bnn_model))

        if len(self.trajectory_median[uav_idx]) == 0:
            reward = min(KL,1)
        else:
            reward = KL/median(self.trajectory_median[uav_idx])
        
        self.trajectory_median[uav_idx].append(KL)
        if len(self.trajectory_median[uav_idx]) > self.config['median_list_length']:
            self.trajectory_median[uav_idx].pop(0)
        return reward
    
    def _terminate_condition(self):
        
        if self.game_steps >= self.horizon:
            return True
        return False

    def _action_1_2_transform(self, in_actions, usage = '2_to_1'):
        
        length = self.config['neighbour_range'] * 2 + 1 
        center_index = self.config['action_dim'] // 2  
        if usage == '1_to_2':
            if in_actions < self.config['action_dim']: 
                i = in_actions // length
                j = in_actions - (in_actions // length) * length
                return (i - self.config['neighbour_range'], j - self.config['neighbour_range']) 
            
            elif in_actions < self.config['action_dim'] + self.config['floor_num']:
                num_fl = int(in_actions - self.config['action_dim'])
                return str(num_fl)
            
            elif in_actions == self.config['action_dim'] + self.config['floor_num']:
                
                return 'superconnect'
            else :
                
                assert True == False  
        elif usage == '2_to_1':
            list_length = self.config['total_action_dim']
            
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
           
        action_transformed = self._action_1_2_transform(action, '1_to_2')
        idx1 = self.legal_pos[pos[0]].index(pos[1]) 

        if action_transformed == 'superconnect':
            if pos[1] in self.super_lists_split[pos[0]]:
                
                return True
            else:
                return False
        elif type(action_transformed) == str:
            
            if self.fl_map[pos[0]][idx1][idx1] == 2 and self.config['floor_list'].index(pos[0]) != int(action_transformed):
                return True
            else:
                return False
        else:
            
            actual_pos = (pos[1][0] + action_transformed[0] * self.config['poi_distance'], 
                pos[1][1] + action_transformed[1] * self.config['poi_distance'])
            try:
                idx2 = self.legal_pos[pos[0]].index(actual_pos)
            except ValueError as e:
                
                return False
            if self.fl_map[pos[0]][idx1][idx2] != 0:
                
                return True
            return False

    def _update_pos(self, uav_index, transformed_action):
        
        
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
            
            elevator_name = self.ele_pos_mapping[self.config['floor_list'][fl_index]][self.uav_pos[uav_index][1]]
            elevator_pos = self.elevator[self.config['floor_list'][transformed_fl_index]][elevator_name]

            
            print('UAV ' + str(uav_index) + ' takes ' + elevator_name + ' on timestep '+ \
                  str(self.game_steps) + ' from ' + str(fl_index) + ' to ' + str(transformed_fl_index))      
            self.uav_pos[uav_index] = (self.config['floor_list'][transformed_fl_index], elevator_pos) 
        else:
            
            new_pos = (self.uav_pos[uav_index][1][0] + transformed_action[0] * self.config['poi_distance'],
               self.uav_pos[uav_index][1][1] + transformed_action[1] * self.config['poi_distance'])
            self.uav_pos[uav_index] = (self.uav_pos[uav_index][0], new_pos) 

    def _data_change_old(self, agent_num, transformed_action):
        maximum_collect = self.config['maximum_collect']
        
        if self.data_amount[self.uav_pos[agent_num][0]][self.uav_pos[agent_num][1]] >= maximum_collect:
            self.data_amount[self.uav_pos[agent_num][0]][self.uav_pos[agent_num][1]] -= maximum_collect
            return maximum_collect
        else:
            collected = self.data_amount[self.uav_pos[agent_num][0]][self.uav_pos[agent_num][1]]
            self.data_amount[self.uav_pos[agent_num][0]][self.uav_pos[agent_num][1]] = 0
            return collected
    
    def _data_change(self, agent_num, transformed_action):
           
        delta = {}
        maximum_collect = self.config['maximum_collect']
        collected = 0
        curr_fl = self.uav_pos[agent_num][0]
        curr_x = self.uav_pos[agent_num][1][0]
        curr_y = self.uav_pos[agent_num][1][1]
        visit_pos_list = []

        if type(transformed_action) == str:
            
            coll = min(maximum_collect, self.data_amount[self.uav_pos[agent_num][0]][self.uav_pos[agent_num][1]] )
            self.data_amount[self.uav_pos[agent_num][0]][self.uav_pos[agent_num][1]] = 0
            
            delta[self.uav_pos[agent_num]] = self.game_steps - self.last_visit[curr_fl][(curr_x,curr_y)]
            self.last_visit[self.uav_pos[agent_num][0]][self.uav_pos[agent_num][1]] = self.game_steps
            self.aoi[self.uav_pos[agent_num][0]][self.uav_pos[agent_num][1]] = 0
            visit_pos_list.append(self.uav_pos[agent_num])
            return coll,delta,visit_pos_list
        
        for x in range(abs(transformed_action[0])):
            for y in range(x, abs(transformed_action[1])):                               
                if y < x:
                    
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
                    
                    self.current_occupancy[curr_fl][combine_pos] = agent_num

                    collected += self.data_amount[curr_fl][combine_pos]
                    self.data_amount[curr_fl][combine_pos] = 0

                    delta[(curr_fl,combine_pos)] = self.game_steps - self.last_visit[curr_fl][combine_pos]
                    self.last_visit[curr_fl][combine_pos] = self.game_steps
                    self.aoi[curr_fl][combine_pos] = 0
                    visit_pos_list.append((curr_fl,combine_pos))

        return collected,delta,visit_pos_list     

    def _masked_action(self):
        
        ret = {fl:{pos:[] for pos in self.legal_pos[fl]} for fl in self.config['floor_list']}
        for fl in self.config['floor_list']:
            for pos in self.legal_pos[fl]:
                for i in range(self.action_dim):
           
                    if self._validate_action(i, (fl,pos)):
                        ret[fl][pos].append(1)
                    else:
                        ret[fl][pos].append(0)
             
        return ret

    def _init_aoi(self):
        
        
        self.aoi = {}
        self.last_visit = {}
        for fl in self.config['floor_list']:
            self.aoi[fl] = {}
            self.last_visit[fl] = {}
            for pos in self.legal_pos[fl]:
                self.aoi[fl][pos] = 0
                self.last_visit[fl][pos] = 0

    def _aoi_default_updates(self):
        
        for fl in self.config['floor_list']:
            for pos in self.legal_pos[fl]:
                self.aoi[fl][pos] += 1

    def _aoi_updates_old(self, transformed_action, agent):
        
        fl = self.uav_pos[agent][0]
        pos = self.uav_pos[agent][1]
        delta = self.game_steps - self.last_visit[fl][pos] 
        self.last_visit[fl][pos] = self.game_steps 
        self.aoi[fl][pos] = 0 
        return delta

    def _aoi_updates(self, transformed_action, agent):
        pass

    def _compute_aoi_violation(self):
           
        num_poi = 0
        total = 0
        if 'use_changing_data':
            
            
            if self.game_steps % self.config['data_last_time'] == self.config['data_last_time'] - 1:
                
                for fl in self.config['floor_list']:
                    for pos in self.poi[fl]:
                        num_poi += 1
                        if self.data_amount[fl][pos] == self.data_amount_init[fl][pos]:
                            total += 1      
                self.aoi_violated_ratio = total/(num_poi)

        return self.aoi_violated_ratio               
                
    def _reward_compute_old(self, agent, transformed_action ,delta_data_amount):
        
        if self.config['rewarding_methods'] == 'default':
              
            default_penalty = -2
       
            extra_reward = 0
            if transformed_action == 'upstairs' or transformed_action == 'downstairs': 
                if self.first_step[agent] > self.agent_num/2: 
                    
                    if transformed_action == 'upstairs':
                        extra_reward = 400
                    print('floor changed with extra rewards ',agent, 'at time step ', self.game_steps)
                    
                    for agentnum in range(self.agent_num):
                        self.first_step[agentnum] -=1
                else:
                    
                    extra_reward = -500
          
            different_floor_reward = 0
            
            
            distance_reward = 0
         
            return default_penalty + delta_data_amount + extra_reward + different_floor_reward + distance_reward


    def _reward_compute(self, agent, transformed_action ,delta_data_amount, 
                        bnn_model = None, current_obs = None, next_obs = None):
        
        if self.config['rewarding_methods'] == 'default':
             
            default_penalty = -2
            ratio = 0
            ele_reward = 0

            if self.config['use_elevator']:

                if self.game_steps % self.config['data_last_time'] == 0:
                    
                    self.elevator_lock = np.zeros((len(self.elevator[list(self.elevator.keys())[0]]), int(self.config['data_changes_num']),\
                            int(self.config['floor_num'])))

                
                if type(transformed_action) == str and transformed_action != 'superconnect':
                    fl = self.uav_pos[agent][0]
                    pos = self.uav_pos[agent][1]
                    fl_index = self.config['floor_list'].index(self.uav_pos[agent][0])
                    transformed_fl_index = int(transformed_action)
                    
                    elevator_name = self.ele_pos_mapping[self.config['floor_list'][fl_index]][self.uav_pos[agent][1]]
                    elevator_idx = list(self.elevator[self.config['floor_list'][transformed_fl_index]]).index(elevator_name)
                    
                    slot = int(min(self.game_steps // self.config['data_last_time'], self.config['data_changes_num']-1))
                    
                
                    if self.elevator_lock[elevator_idx][slot][transformed_fl_index] == 0:
                        
                        
                        if self.elevator_usage_time[agent] == 0 or abs(self.elevator_usage_time[agent] - self.game_steps) > self.config['data_last_time']/2 :
                            ele_reward = self._determin_elevator_reward(agent)[fl][self.ele_pos_mapping[fl][pos]][transformed_action]
                            self.elevator_lock[elevator_idx][slot][transformed_fl_index] = 1
                        else:
                            ele_reward = -40
                    else:
                        
                        ele_reward = -40
                        

                    if self.game_steps > self.config['horizon'] - 20:
                        
                        ele_reward = 0

                    self.elevator_usage_time[agent] = self.game_steps
                        
            return default_penalty + delta_data_amount + ele_reward * ratio



    def _elevator_DP_reward(self,obs):
        
        pass
    
    def _aoilambda(self,dummy):
  
        print(self.aoi_lambda[0][('site2', 'F2')][(580, 230)])
        print(self.aoi_lambda[0][('site2', 'F3')][(580, 230)])
        print(self.aoi_lambda[0][('site2', 'F4')][(580, 230)])
        return self.aoi_lambda
    
    def _init_lambda_aoi(self):

        self.aoi_lambda = {i:{fl:{pos:0 if pos not in self.poi[fl] else np.random.rand() \
            for pos in self.legal_pos[fl]} for fl in self.config['floor_list']} for i in range(self.config['data_changes_num'])}
        
    def _lambda_projection(self,lambda_list, default_upperbound = float('inf')):
        
        A = np.identity(len(lambda_list)) 
        b = np.array(lambda_list)
        s = np.array([default_upperbound for _ in range(len(lambda_list))])
        x = cp.Variable(len(lambda_list))
        prob = cp.Problem(cp.Minimize(cp.sum_squares(A@x - b)),[default_upperbound >= x, x >= 0])
        prob.solve()
 
        return np.round(x.value,3)

    def _lambda_projection_KKT(self,lambda_list,default_upperbound = float('inf')):
        
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
        ret_rewards = agent_rewards     
        for pos in visit_pos_list:
            ret_rewards += self.aoi_lambda[pos[0]][pos[1]] * min(self.aoi[pos[0]][pos[1]], self.config['dual_clip_value'])

        return ret_rewards

    def _compute_lambda_updates(self, constraint_dict):
        for i in range(self.config['data_changes_num']):
            for fl in self.config['floor_list']:
                for pos in self.poi[fl]:
                    self.aoi_lambda[i][fl][pos] -= self.config['dual_descent_learning_rate'] * constraint_dict[i][fl][pos]
        
        
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

        d_default_2 = 500
        
        rate = 0.003
        total_d_2 = 0

        for i in range(self.agent_num):
            if self.uav_pos[i][0] == self.uav_pos[agent][0]:
                
                total_d_2 += (self.uav_pos[i][1][0] - self.uav_pos[agent][1][0])**2 \
                    + (self.uav_pos[i][1][1] - self.uav_pos[agent][1][1])**2
            else:
                
                total_d_2 +=  d_default_2
        return np.sqrt(total_d_2) * rate
    
    def _compute_total_data(self, mode = 'init', period = None):
        
        total = 0
        if mode == 'init':
            for fl in self.config['floor_list']:
                for pos in self.legal_pos[fl]:
                    total += self.data_amount_init[fl][pos]
        elif mode == 'init_with_changes':
            
            for fl in self.config['floor_list']:
                for pos in self.legal_pos[fl]:
                    total += self.data_amount_init[fl][pos]
        elif mode == 'runtime':
            for fl in self.config['floor_list']:
                for pos in self.legal_pos[fl]:
                    total += self.data_amount[fl][pos]
        return total

    def _init_data_amount(self):
        
        if self.config['use_changing_data']:
            data = 0
            for i in range(self.config['data_changes_num']):
                data += self._compute_total_data('init_with_changes', period=i) 
        else:
            data = self._compute_total_data('init')
        return data
        
    

if __name__ == '__main__':
    env = EnvCore()
