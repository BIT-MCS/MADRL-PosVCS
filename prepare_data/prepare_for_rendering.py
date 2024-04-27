import cv2 as cv
import numpy as np
import pickle
from copy import deepcopy

def _generate_poi_jpg():
    #生成一个像素点，储存，作为POI
    ret = np.zeros([1,1,3])
    ret[0][0] = np.array([255,0,0])
    cv.imwrite('./poi_render.png',ret)

def take_away_dict_postprocess(in_dict):
    # 从已经被处理过后的dict中整理成喂给render的格式
    # in_dict key: 'observation', 'rewards','dones','info' [#envs * agents * dim]
    # env指的是开了多少个进程环境 这里default 5，agents是无人机数目

    obs = in_dict['observation']
    rewards = in_dict['rewards']
    dones = in_dict['dones']
    info = in_dict['info'] # list of dicts, keys for a dict : 'current_position' ,'transformed_action',''pos_info','available_actions','collection_ratio'
    
    num_agent = len(obs[0])
    episode = len(obs)
    keys = ['reward_history', 'uav_trace', 'poi_history','uav_reward','task_reward']
    ret = {} 
   
    '''
    有关格式这里必须特殊说明下，以最复杂的info为例
    >>> len(in_dict('info')) # 这个长度是一个episode里边step长度
    250 

    >>> len(in_dict('info')[0]) # 这个长度是第一个时间步里，subenv的个数
    5

    >>> len(in_dict('info')[0][0]) # 这个长度是第一个时间步，第一个subenv里的agent数
    2

    >>> len(in_dict('info')[0][0][0]) # 这个长度是第一个时间步，第一个subenv里，第一个agent里字典key的长度
    3

    >>> in_dict['info'][0][0][0].keys() # 这个是第一个时间步，第一个subenv里，第一个agent里字典key
    dict_keys(['current_position', 'transformed_action', 'pos_info'])

    >>> len(in_dict['info'][0][0][0]['pos_info']) # 这个长度是第一个时间步，第一个subenv里，第一个agent里字典'pos_info' 里的字典的key长度
    2

    >>> in_dict['info'][0][0][0]['pos_info'].keys() # 这个是第一个时间步，第一个subenv里，第一个agent里'pos_info'键下的keys
    dict_keys(['pos', 'val'])

    >>> d['info'][0][0][0]['pos_info'] == d['info'][0][0][1]['pos_info'] #同一个时间，同一个subenv下 不同agent的pos info是一致的，
    TODO： 考虑删除多余
    True
    '''

    # uav_trace
    # 格式：[[(('site2', 'F3'), (580, 620)), (('site2', 'F3'), (580, 620)),...], [...], [...], ...]
    uav_trace = []
    for agent in range(num_agent):
        trajectory = []
        for i in range(episode):
            # 第一个0是选取多个thread中的第一个,第二个0是套着list呢
            # 处理完后是一个dict， 示例{'current_position': (('site2', 'F3'), (580, 620)), 'transformed_action': (0, 2)}
            
            # 需要对坐标处理下
            # copydict = deepcopy(info[i][0][agent])
            # copydict['current_position'] = (copydict['current_position'][0], _coordinate_change_after_rendering(
            #      list(info[i][0][agent]['pos_info']['pos'].keys()).index(copydict['current_position'][0]),
            #      copydict['current_position'][1],900))
            # trajectory.append(copydict)
            old_pos = info[i][agent]['current_position']
            #trajectory.append(info[i][0][agent]['current_position'])
            trajectory.append((old_pos[0], _coordinate_change_after_rendering_2(
                list(info[i][agent]['pos_info']['pos'].keys()).index(old_pos[0]), old_pos[1], 900, 1035)))
        uav_trace.append(trajectory)
    ret['uav_trace'] = uav_trace

    #reward_history
    # 格式：[[75,-1,-1,...], [...], [...],...]

    r = []
    for agent in range(num_agent):
        self_r = []
        for i in range(episode):
            self_r.append(rewards[i][agent])
        r.append(self_r)
    ret['reward_history'] = r

    #poi_history
    # 格式：[{'pos':[], 'val':[],}, {'pos':[], 'val':[]},...] 长度为episode length
    pois = []
    for i in range(episode):
        i_poi = {}
        for agent in range(num_agent):
            i_poi['pos'] = _transform(info[i][agent]['pos_info']['pos'])
            i_poi['val'] = info[i][agent]['pos_info']['val']
        pois.append(i_poi)
    ret['poi_history'] = pois

    # collection_ratio
    # 格式：[[0.01,0.02,0.04,...], [...], [...],...]
    data_ratio = []
    for agent in range(num_agent):
        self_ratio = []
        for i in range(episode):
            self_ratio.append(info[i][agent]['collection_ratio'])
        data_ratio.append(self_ratio)
    ret['collection_ratio'] = data_ratio

    # violation_ratio
    # 格式：[0,0,0,...,0,3,.,.]
    violation_ratio = []
    for i in range(episode):
        violation_ratio.append(info[i][0]['violation_ratio'])
    data_ratio.append(violation_ratio)
    ret['violation_ratio'] = violation_ratio 
    
    #uav_reward
    
    #task_reward

    return ret

def _concatenate(dir_list):
    # 从dir_list中读出位置, 并存储处理后的图片格式为 concatenate_for_render.png
    # list 中还是从楼层低到楼层高处理
    img_list = []
    for dir in dir_list:
        img_list.append(cv.imread(dir))
    vis = img_list[0]
    if len(dir_list) > 1:    
        for i in range(1,len(dir_list)):
            vis = np.concatenate((vis,img_list[i]), axis=1)
    print(vis.shape)
    cv.imwrite('./intermediate/pictures/concatenate_for_render.png', vis)
    return vis

def _concatenate_2(dir_list):
    # 从dir_list中读出位置, 并存储处理后的图片格式为 concatenate_for_render.png
    # list 中还是从楼层低到楼层高处理
    img_list = []
    for dir in dir_list:
        img_list.append(cv.imread(dir))

    vis1 = img_list[0]
    vis2 = img_list[3]
    for i in range(1,3):
        vis1 = np.concatenate((vis1,img_list[i]), axis=1)

    for i in range(1,3):
        vis2 = np.concatenate((vis2,img_list[3+i]), axis=1)
    
    vis_all = np.concatenate((vis1,vis2), axis = 0)
    print(vis_all.shape)
    cv.imwrite('./intermediate/pictures/concatenate2_for_render.png', vis_all)
    return vis_all

def _coordinate_change_after_rendering(floor_idx, pos, horizontal):
    # floor_idx 是floor_list中的index数
    return (floor_idx * horizontal + pos[1], pos[0])

def _coordinate_change_after_rendering_2(floor_idx, pos, horizontal, vertical):
    # floor_idx 是floor_list中的index数
    if floor_idx > 2:
        return ((floor_idx - 3) * horizontal + pos[1], pos[0] + vertical)
    else:
        return (floor_idx * horizontal + pos[1], pos[0])

def _transform(in_dict):
    # in_dict: {('site2','F3'):[],...}
    # 对position进行坐标转化
    ret = {}
    for i in range(len(list(in_dict.keys()))):
        k = list(in_dict.keys())
        ret[k[i]] = []
        for index in in_dict[k[i]]:
            #ret[k[i]].append(_coordinate_change_after_rendering(i, index, 900)) #TODO: 写进config        
            ret[k[i]].append(_coordinate_change_after_rendering_2(i, index, 900, 1035)) #TODO: 写进config
    return ret

if __name__ == "__main__":
    with open('./intermediate/data/generate_dict.pickle', 'rb') as f:
        in_dict = pickle.load(f)
    take_away_dict_postprocess(in_dict) 
    _concatenate(['./prepare_data/output/mod_floorplan/site3/F1_POI_on_grid.png',\
                  './prepare_data/output/mod_floorplan/site3/F2_POI_on_grid.png','./prepare_data/output/mod_floorplan/site3/F3_POI_on_grid.png'])
