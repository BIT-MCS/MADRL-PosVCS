# 用简单的TSP构建一个Baseline
# 无人机均匀分布

import numpy as np
from python_tsp.exact import solve_tsp_dynamic_programming
from python_tsp.heuristics import solve_tsp_simulated_annealing
import sys
import os
import heapq
sys.path.append(os.getcwd())
sys.path.append(os.getcwd() + '/MAPPO')
from TSP.tsp_config import config
import pickle
from MAPPO.envs.env_core import EnvCore
import networkx as nx

from prepare_for_rendering import _coordinate_change_after_rendering, _transform

def tsp_return():

    # 第一步，均匀规划无人机的起点

    agent_num = config['agent_num']
    start_pos = []

    env = EnvCore()

    # 将信息load进来
    with open('./states_info.pkl', 'rb') as f:
        myinstance = pickle.load(f)
        for k in myinstance.__dict__.keys():
            setattr(env, k, getattr(myinstance, k))

    # 加入各个无人机位置 初始的地图位置 
    #start_pos.append(env.config['uav_init_pos'][1])
    # 
    start_pos.extend([list(env.elevator[env.config['uav_init_pos'][0]].values())[i] for i in range(len(env.elevator[env.config['uav_init_pos'][0]]))])

    assert env.config['agent_num'] <= len(start_pos)
    start_pos = start_pos[:env.config['agent_num']]

    # 第二步，给每个无人机分个subgraph，每个无人机找到最近的n部分
    # 这部分就简单的将POI均分给每个无人车，然后所有无人车从起点先移动到tsp路径起点然后绕圈

    d = lambda x,y: np.sqrt((x[0]-y[0])**2 + (x[1]-y[1])**2)

    subgraphs = {starts:[(fl,list(env.elevator[fl].values())[list(env.elevator[env.config['uav_init_pos'][0]].values()).index(starts)]) \
                        for fl in env.config['floor_list']] for starts in start_pos} # {起始点：[(fl,pos),(fl,pos)],...}
    total_poi_num = sum([1 for fl in env.config['floor_list'] for poi in env.poi[fl]])

    for fl in env.config['floor_list']:
        for poi in env.poi[fl]:
            min_d = float('inf')
            point = start_pos[0]
            for pos in start_pos[1:]:
                # 均匀分配
                if d(pos, point) < min_d and len(subgraphs[pos]) <= total_poi_num / env.config['agent_num'] :
                    min_d = d(pos, point)
                    point = pos
            subgraphs[point].append((fl,poi))

    # 第三步，对每个子集跑一个tsp，对比收集率和违犯AOI的个数，如果没有够时间数就继续绕圈

    tsp_graph = {}
    d_mah = lambda x,y: abs(x[0] - y[0]) + abs(x[1] - y[1])

    #构建距离矩阵：
    for starts in subgraphs.keys():
        all_starts = [(fl,starts) for fl in env.config['floor_list']]
        tsp_graph[starts] = nx.Graph() # 这个图用0，1，2，3，...,来表示
        tsp_graph[starts].add_nodes_from(list(range(len(subgraphs[starts]))))
        for idx1 in range(len(subgraphs[starts])):
            for idx2 in range(len(subgraphs[starts])):
                p1 = subgraphs[starts][idx1]
                p2 = subgraphs[starts][idx2]
                if p1[0] != p2[0]:
                    # 不在一层，先不联通
                    continue
                else:
                    tsp_graph[starts].add_edge(idx1,idx2)
                    tsp_graph[starts][idx1][idx2]['weight'] = d_mah(p1[1],p2[1])


        # 将不连接的部分连接上
        for i1 in range(len(all_starts)):
            for i2 in range(len(all_starts)):
                tsp_graph[starts].add_edge(i1,i2)
                tsp_graph[starts][i1][i2]['weight'] = 1
        
        
    # 模拟退火算个近似解
    patrolling_seq = {}
    for starts in subgraphs.keys():
        tsp = nx.approximation.traveling_salesman_problem
        patrolling_seq[starts] = tsp(tsp_graph[starts])

    #生存render需要的路径

    # 先从起点走到指定地点
    # 注意现在需要在真实图上找出最近的点了
    true_graph = {fl:nx.Graph() for fl in env.config['floor_list']}
    for fl in env.config['floor_list']:
        adj = env.fl_map[fl]
        edge_list = []
        for i in range(adj.shape[0]):
            for j in range(adj.shape[1]):
                if adj[i][j] != 0:
                    edge_list.append((i,j,d_mah(env.legal_pos[fl][i],env.legal_pos[fl][j])))
        true_graph[fl].add_weighted_edges_from(edge_list)

    step_seq = {} # 这个是真实环境中的点列，index对应legal_pos
    uav_init_pos = env.config['uav_init_pos']
    for starts in subgraphs.keys():
        s = env.legal_pos[uav_init_pos[0]].index(uav_init_pos[1])
        e = env.legal_pos[uav_init_pos[0]].index(starts)
        path = nx.shortest_path(true_graph[uav_init_pos[0]], s, e)
        step_seq[starts] = list(zip([uav_init_pos[0] for _ in range(len(path))],[env.legal_pos[uav_init_pos[0]][path[i]] for i in range(len(path))]))
        step_left = env.config['horizon'] - len(path)
        step_count = 0
        # 然后在gamestep的剩余时间里绕圈
        idxs = 0
        while True:
            actual_seq = [subgraphs[starts][patrolling_seq[starts][i]] for i in range(len(patrolling_seq[starts]))]
            
            if actual_seq[idxs % len(patrolling_seq[starts])][0] != actual_seq[(idxs+1) % len(patrolling_seq[starts])][0]:
                # 在上下樓梯
                assert actual_seq[idxs % len(patrolling_seq[starts])][1] == actual_seq[(idxs+1) % len(patrolling_seq[starts])][1]
                idxs += 1 
                fl_before = actual_seq[(idxs) % len(patrolling_seq[starts])][0]
                fl_after = actual_seq[(idxs+1) % len(patrolling_seq[starts])][0]
                step_seq[starts].append((fl_after, list(env.elevator[fl_after].values())\
                            [list(env.elevator[fl_before].values()).index(actual_seq[idxs % len(patrolling_seq[starts])][1])]))
                continue
            s = env.legal_pos[actual_seq[idxs % len(patrolling_seq[starts])][0]].index(actual_seq[idxs % len(patrolling_seq[starts])][1])
            e = env.legal_pos[actual_seq[(idxs+1) % len(patrolling_seq[starts])][0]].index(actual_seq[(idxs+1) % len(patrolling_seq[starts])][1])
            
            path = nx.shortest_path(true_graph[actual_seq[idxs % len(patrolling_seq[starts])][0]], s, e)
            step_count += len(path) - 1
            step_seq[starts].extend([(actual_seq[idxs % len(patrolling_seq[starts])][0], 
                            env.legal_pos[actual_seq[idxs % len(patrolling_seq[starts])][0]][item]) for item in path[1:]])
            idxs += 1

            if step_count > step_left:
                break

    # for keys in step_seq.keys():
    #     print(step_seq[keys], '\n')

    # 根据step_seq 选择动作列，喂入环境
    action_seq = {}
    for starts in step_seq.keys():
        action_seq[starts] = []
        for i in range(len(step_seq[starts])-1):
            s0 = step_seq[starts][i]
            s1 = step_seq[starts][i+1]
            dx = (s1[1][0] - s0 [1][0])//env.config['poi_distance']
            dy = (s1[1][1] - s0[1][1])//env.config['poi_distance']
            if s0[0] != s1[0]:
                #print(env.config['floor_list'].index(s0[0]),env.config['floor_list'].index(s1[0]))
                
                if env.config['floor_list'].index(s0[0]) < env.config['floor_list'].index(s1[0]):
                    transformed_action = 'upstairs'
                else:
                    transformed_action = 'downstairs'
            elif abs(dx) > 2 or abs(dy) > 2:
                # superaction
                assert env.fl_map[s0[0]][env.legal_pos[s0[0]].index(s0[1])][env.legal_pos[s0[0]].index(s1[1])] == 1
                transformed_action = 'superconnect'
            else:
                transformed_action = (dx,dy)
            action_idx = env._action_1_2_transform(transformed_action)
            action_seq[starts].append([1 if i == action_idx else 0 for i in range(env.config['total_action_dim'])])
    
    #重置環境        
    env.reset()

    ret = {}
    '''
    len(info['uav_trace']) = 5
    len(info['uav_trace'][0]) = 250
    info['uav_trace'][0] = (('site2','F3'),(580, 580))
    len(info['reward_history'][0]) = 250
    info['reward_history'][0] = array([-2])


    '''
    ret['uav_trace'] = [[] for i in range(env.agent_num)] 
    ret['reward_history'] = [[np.array([0])] for i in range(env.agent_num)] 
    ret['poi_history'] = [{} for i in range(env.config['horizon'])]
    ret['collection_ratio'] = [[] for i in range(env.agent_num)]
    ret['violation_ratio'] = [] 
    for i in range(env.config['horizon']):
        action_step = []
        for agents_id in range(len(action_seq.keys())):
            p = step_seq[list(action_seq.keys())[agents_id]][i]
            ret['uav_trace'][agents_id].append(_coordinate_change_after_rendering(env.config['floor_list'].index(p[0]), p[1], 900))
            action_step.append(action_seq[list(action_seq.keys())[agents_id]][i])

        # [sub_agent_obs, sub_agent_reward, sub_agent_done, sub_agent_info]
        returned_from_step = env.step(action_step, rend = True)

        for agents_id in range(len(action_seq.keys())):
            ret['reward_history'][agents_id].append(returned_from_step[1])
            ret['collection_ratio'][agents_id].append(env.take_out_dict['collection_ratio'])
        ret['violation_ratio'].append(env.take_out_dict['violation_ratio'])
        ret['poi_history'][i]['pos'] = _transform(env.take_out_dict['pos_info']['pos'])
        ret['poi_history'][i]['val'] = env.take_out_dict['pos_info']['val']
    print('collection ratio upto step ' + str(i) + ' is ' + str(np.round(env.runtime_data/env.total_data, 3)))

    return ret
    
if __name__ == '__main__':
    tsp_return()