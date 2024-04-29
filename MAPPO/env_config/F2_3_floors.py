config = {
'poi_distance': 10, 
'neighbour_range': 2, 
'action_dim': 25, 
'elevator_action_dim': 3, 
'super_action_dim': 1, 
'total_action_dim': 29, 
'use_global_observation_only': False , 
'observation_range': 5, 
'up_down_range': 3, 
'agent_num': 5, 
'floor_num': 3,
'floor_list': [('site2', 'F2'), ('site2', 'F3'), ('site2', 'F4')],
'top_floor':('site2', 'F4'),
'bottom_floor':('site2', 'F2'),
'horizon': 250, 
'maximum_collect': 175, 
'render_collect' :175, 
'noise_level': 5, 
'observation_dim':None,
'elevator_pos':None,
'transition_mode':['full_success','with_random_collision', \
        'with_deterministic_collision','with_malicious_and_deterministic_collision'][0], 
'malicious_exist': False, 
'use_malicious_evaluate': False, 
'use_masked_action': True, 
'if_read_states': False, # generate new states info each time if environment change, otherwise False.
'if_read_states_besides_config':True, 
'uav_energy':1,
'AP_units':20, 
'uav_init_pos': (('site2', 'F3'), (580, 600)),
'uav_init_pos_2': (('site2', 'F3'), (580, 600)),
'rewarding_methods':'default', 
'use_dual_descent': True, 
'dual_descent_learning_rate': 0.05,
'dual_upperbound': 5,
'use_vime': True,
'median_list_length': 40, 
'vime_reward_ratio': 20 , 
'use_elevator': False,
'use_changing_data': True, 
'data_changes_num': 2, 
'use_same_data': True, 
'data_last_time': 125, 
'dual_clip_value': 125, 
'aoi_penalty_bound': 166 
}


 
