config = {
# 环境参数
'poi_distance': 10, #基础单位，代表两个poi在图像间差几个像素
'neighbour_range': 2, # 2个点单位的L1 ball，也即20像素内的L1 ball
'action_dim': 25, # 和上边对应
'elevator_action_dim': 2, # 电梯处多的动作，上楼和下楼
'total_action_dim': 27, # 25 + 2
'use_global_observation_only': False , #是否喂全局信息，全局喂的时候需要注意智能体间的obs可能很类似
'observation_range': 5, # 观测范围，范围外的全mask掉
'up_down_range': 3, # 半径为3的L1区域内都可以有上下楼的动作
'agent_num': 4, #TODO:对于大于2个agent的情况 还需要和代码对接 
'floor_num': 2,
'floor_list': [('site2', 'F3'), ('site2', 'F4')],
'top_floor':('site2', 'F4'),
'bottom_floor':('site2', 'F3'),
'horizon': 250, # 记得要在MAPPO里的算法里同样改了 现在是250
'maximum_collect': 75, # 无人车最大一步收集能力
'render_collect' :75, # 实测时候的收集能力
'noise_level': 5, # 实测数据噪音上下界
'observation_dim':None,
'elevator_pos':None,
'transition_mode':['full_success','with_random_collision', \
        'with_deterministic_collision','with_malicious_and_deterministic_collision'][0], #转移概率，这里换数字切换模式就好
'malicious_exist': True, # 是否用环境干扰
'use_masked_action': True, # 是否将撞墙的信息masked掉
'if_read_states': True, # 是否读入预存的states信息，如果要更新states信息记得False掉，重新存一遍
'if_read_states_besides_config':True, # 是否用当前的config，而不是load的config，这个在只想简化poi和图片计算用

# UAV参数
'uav_energy':1,
'AP_units':20, # 代表多少个AP算是一份时间内收集的, 也即1000个AP就有50份数据量
'uav_init_pos': (('site2', 'F3'), (580, 600)),

#奖励参数
'rewarding_methods':'default', # default 为基本设置

#算法参数
'use_dual_descent': True, # 是否使用对偶梯度
# TODO: 使用对偶梯度的时候貌似会导致
'dual_descent_learning_rate': 0.001,

#数据量和算法参数交叉部分
'use_changing_data': True, # 是否使数据按时间变化
'data_changes_num': 3, # 代表全局poi数据量刷新几次
'use_same_data': True, # 每次数据刷新是否用一样的数据，False的话需要去重新生成多个GP来预测, default:True
'data_last_time': 83, # 'horizon'/ 'data_changes_num',表示数据一次的持续时间
'dual_clip_value': 83, # 这个值应该设计为和data一次的持续时间时长一样， 'horizon'/ 'data_changes_num'
'aoi_penalty_bound': 166 # 这个代表希望poi的aoi在哪一个上限内，目前对于所有poi都是一致的



}


 
