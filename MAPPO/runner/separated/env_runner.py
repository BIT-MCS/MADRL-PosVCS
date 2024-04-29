import time
import wandb
import os
import numpy as np
from itertools import chain
import torch
import pickle

from utils.util import update_linear_schedule
from runner.separated.base_runner import Runner
import imageio
import VI

def _t2n(x):
    return x.detach().cpu().numpy()


class EnvRunner(Runner):
    def __init__(self, config):
        super(EnvRunner, self).__init__(config)


    def run(self,device):
        self.warmup()

        start = time.time()
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads

        
        rend = False
        if self.all_args.generate_outputs:
            ret_dict = {'observation':[],'rewards':[],'dones': [],'info':[]}
            episodes = 1 
            rend = True

        if self.envs.env_ref.env.config['use_dual_descent']:
            agent_num = self.envs.env_ref.env.config['agent_num']
            legal_pos = self.envs.env_ref.env.legal_pos
            poi_pos = self.envs.env_ref.env.poi
            floor_list = self.envs.env_ref.env.config['floor_list']
            time_interval = self.envs.env_ref.env.config['data_changes_num']
        
        
        transition_prob_model = VI.build_network(32,32,device)
        
        
        vime_pool = VI.SimpleReplayPool(2000,(32,),self.envs.env_ref.env.config['total_action_dim'])
        
        for episode in range(episodes):
            if self.use_linear_lr_decay:
                for agent_id in range(self.num_agents):
                    self.trainer[agent_id].policy.lr_decay(episode, episodes)

            previous_obs = self.envs.env_ref.env.initial_obs
            trajectory_median = [[] for _ in range(self.envs.env_ref.env.config['agent_num'])]
            
            
            d_list = []

            
            track_visit = {en:{i:{uv:{fl:{pois:0 for pois in poi_pos[fl]} for fl in floor_list} for uv in range(agent_num)} \
                           for i in range(time_interval)} for en in range(self.envs.nenvs)}
            track_all = {en:{i:{fl:{pois:0 for pois in poi_pos[fl]} for fl in floor_list} for i in range(time_interval)}for en in range(self.envs.nenvs)}

            obs_list = []
            action_list = []

            for step in range(self.episode_length):
                
                values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env = self.collect(step)

                
                obs, rewards, dones, infos = self.envs.step(actions_env, rend)
                obs_list.append(obs)
                action_list.append(np.array(actions_env))

                
                if self.envs.env_ref.env.config['use_dual_descent'] and not rend:
                    period = int(step * self.envs.env_ref.env.config['data_changes_num'] /self.episode_length)
                    if self.envs.env_ref.env.config['use_dual_descent']:
                        for envs in range(self.envs.num_envs):
                            for uav in range(self.envs.env_ref.env.config['agent_num']):
                                for item in infos[envs][uav]['aoi_reduce'].items():
                                    pos = item[0]      
                                    if pos[1] in poi_pos[pos[0]]:
                                        track_visit[envs][period][uav][pos[0]][pos[1]] = 1
                                        track_all[envs][period][pos[0]][pos[1]] += 1
                
                
                if self.all_args.generate_outputs:
                    ret_dict['observation'].append(obs[0])
                    ret_dict['rewards'].append(rewards[0])
                    ret_dict['dones'].append(dones[0])
                    ret_dict['info'].append(infos[0]) 

                data = obs, rewards, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic
                d_list.append(data)

                
                self.insert(data)

            
            modified_rewards = np.zeros([rewards.shape[1],len(d_list),self.envs.nenvs,1]) 
            for agents in range(rewards.shape[1]):
                    for i in range(len(d_list)):                      
                        for num_envs in range(rewards.shape[0]):      
                            modified_rewards[agents,i,num_envs,0] = d_list[i][1][num_envs,agents,0]

            if self.envs.env_ref.env.config['use_dual_descent'] and not rend:
                period_reward = {}
                for num_envs in range(self.envs.num_envs):
                    lambdalist = self.envs.aoilambda(index = num_envs)
                    count_adding = 0
                    for step in range(len(d_list)):
                        interval = int(step * self.envs.env_ref.env.config['data_changes_num'] /self.episode_length)
                        if interval not in period_reward.keys():
                            
                            period_reward[interval] = np.zeros([len(d_list),self.envs.num_envs,1])
                        for agent in range(agent_num):
                            for fl in floor_list:
                                for pos in poi_pos[fl]:
                                    append_reward = 5 * 1/self.envs.env_ref.env.config['data_last_time'] \
                                        * lambdalist[interval][fl][pos] * track_visit[num_envs][interval][agent][fl][pos]
                                    period_reward[interval][step,num_envs,0] += append_reward
                                    count_adding += append_reward
                    print('This env add total rewards: ', count_adding)

                for agents in range(rewards.shape[1]):
                    for i in range(len(d_list)):                      
                        for num_envs in range(rewards.shape[0]):      
                            modified_rewards[agents,i,num_envs,0] += period_reward[interval][i,num_envs,0]
                
                deltavis_dict = {envs:{i:{fl:{pos:0 for pos in poi_pos[fl]} for fl in floor_list} \
                                    for i in range(self.envs.env_ref.env.config['data_changes_num'])} for envs in range(self.envs.nenvs)}
                for envs in range(self.envs.nenvs):
                    for i in range(self.envs.env_ref.env.config['data_changes_num']):
                        for fl in floor_list:
                            for pos in poi_pos[fl]:
                                deltavis_dict[envs][i][fl][pos] = - 1
                                for num_uv in range(agent_num):
                                    if track_visit[envs][i][num_uv][fl][pos] == 1:
                                        deltavis_dict[envs][i][fl][pos] = track_all[envs][i][fl][pos] - 1
                                        break
                
                for envs in range(self.envs.num_envs):
                    curr_env = self.envs._compute_lambda_updates(deltavis_dict[envs], index = envs)

            if self.envs.env_ref.env.config['use_vime'] and not rend:
            
                trajectory_median = [[] for _ in range(self.envs.env_ref.env.config['agent_num'])]
                new_obslist = {}
                for i in range(len(obs_list)):
                    obs = obs_list[i]
                    act = action_list[i]

                    for x in range(obs.shape[0]): 
                        for y in range(obs.shape[1]): 
                            newobs = self.envs._vime_obs_process(obs[x,y,:],act[x,y,:],index = x) 
                            new_obslist[(x,y)] = newobs
                            vime_pool.add_sample(newobs[y,:],actions[x,y,:],rewards[x,y,:],dones[x,y])
                    if i > 0:
                        for num_envs in range(rewards.shape[0]):
                            for agents in range(rewards.shape[1]):       
                                vime_reward, trajectory_median = VI.determine_vime_reward(
                                    (agents, transition_prob_model, torch.from_numpy(new_obslist[num_envs,agents]), \
                                        torch.from_numpy(new_obslist[num_envs,agents]), trajectory_median))    
                                modified_rewards[agents,i,num_envs,0] +=  vime_reward 
            
            self.update_rewards(modified_rewards)
            md = VI.convert_to_mean_nn(transition_prob_model)
            #print(torch.round(md(torch.tensor(self.envs._vime_obs_process(obs_list[0][0,0,:],action_list[0][0,0,:],index = 0)).float())))
            
            train_infos = self.train()

            
            if self.envs.env_ref.env.config['use_vime']:
                VI.compute(transition_prob_model, vime_pool)

            
            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads

            
            if (episode % self.save_interval == 0 or episode == episodes - 1):
                self.save()

            
            if episode % self.log_interval == 0:
                end = time.time()
                print("\n Scenario {} Algo {} Exp {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}.\n"
                      .format(self.all_args.scenario_name,
                              self.algorithm_name,
                              self.experiment_name,
                              episode,
                              episodes,
                              total_num_steps,
                              self.num_env_steps,
                              int(total_num_steps / (end - start))))

                
                for agent_id in range(self.num_agents):
     
                    train_infos[agent_id].update(
                        {"average_episode_rewards": np.mean(self.buffer[agent_id].rewards) * self.episode_length})
                    print("episode rewards for agent {} is {}".format(agent_id, train_infos[agent_id]["average_episode_rewards"]))
                
                self.log_train(train_infos, total_num_steps)
            
                
                if episode % (10 * self.log_interval):
                    self.envs.env_ref.env._runtime_summary()

            
            if episode % self.eval_interval == 0 and self.use_eval:
                self.eval(total_num_steps)

        if self.all_args.generate_outputs:
        
            try:
                os.remove('./intermediate/data/generate_dict.pickle')
            except OSError as e:
                pass

            with open('./intermediate/data/generate_dict.pickle','wb') as f:
                pickle.dump(ret_dict, f)

    def warmup(self):
        
        obs = self.envs.reset()

        share_obs = []
        for o in obs:
            share_obs.append(list(chain(*o)))
        share_obs = np.array(share_obs)

        for agent_id in range(self.num_agents):
            if not self.use_centralized_V:
                share_obs = np.array(list(obs[:, agent_id]))
            self.buffer[agent_id].share_obs[0] = share_obs.copy()
            self.buffer[agent_id].obs[0] = np.array(list(obs[:, agent_id])).copy()

    @torch.no_grad()
    def collect(self, step):
        values = []
        actions = []
        temp_actions_env = []
        action_log_probs = []
        rnn_states = []
        rnn_states_critic = []

        for agent_id in range(self.num_agents):
            self.trainer[agent_id].prep_rollout()
            value, action, action_log_prob, rnn_state, rnn_state_critic \
                = self.trainer[agent_id].policy.get_actions(self.buffer[agent_id].share_obs[step],
                                                            self.buffer[agent_id].obs[step],
                                                            self.buffer[agent_id].rnn_states[step],
                                                            self.buffer[agent_id].rnn_states_critic[step],
                                                            self.buffer[agent_id].masks[step],
                                                            available_actions = self.buffer[agent_id].available_actions[step])
            
            
            values.append(_t2n(value))
            action = _t2n(action)
            
            if self.envs.action_space[agent_id].__class__.__name__ == 'MultiDiscrete':
                for i in range(self.envs.action_space[agent_id].shape):
                    uc_action_env = np.eye(self.envs.action_space[agent_id].high[i] + 1)[action[:, i]]
                    if i == 0:
                        action_env = uc_action_env
                    else:
                        action_env = np.concatenate((action_env, uc_action_env), axis=1)
            elif self.envs.action_space[agent_id].__class__.__name__ == 'Discrete':
                action_env = np.squeeze(np.eye(self.envs.action_space[agent_id].n)[action], 1)
            else:
                
                action_env = actions
                

            actions.append(action)
            temp_actions_env.append(action_env)
            action_log_probs.append(_t2n(action_log_prob))
            rnn_states.append(_t2n(rnn_state))
            rnn_states_critic.append(_t2n(rnn_state_critic))

        
        actions_env = []
        for i in range(self.n_rollout_threads):
            one_hot_action_env = []
            for temp_action_env in temp_actions_env:
                one_hot_action_env.append(temp_action_env[i])
            actions_env.append(one_hot_action_env)

        values = np.array(values).transpose(1, 0, 2)
        actions = np.array(actions).transpose(1, 0, 2)
        action_log_probs = np.array(action_log_probs).transpose(1, 0, 2)
        rnn_states = np.array(rnn_states).transpose(1, 0, 2, 3)
        rnn_states_critic = np.array(rnn_states_critic).transpose(1, 0, 2, 3)

        return values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env

    def insert(self, data):
        obs, rewards, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic = data

        rnn_states[dones == True] = np.zeros(((dones == True).sum(), self.recurrent_N, self.hidden_size),
                                             dtype=np.float32)
        rnn_states_critic[dones == True] = np.zeros(((dones == True).sum(), self.recurrent_N, self.hidden_size),
                                                    dtype=np.float32)
        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)

        share_obs = []
        for o in obs:
            share_obs.append(list(chain(*o)))
        share_obs = np.array(share_obs)

        for agent_id in range(self.num_agents):
            if not self.use_centralized_V:
                share_obs = np.array(list(obs[:, agent_id]))
            
            
            if self.envs.env_ref.env.config['use_masked_action']:
                masked = np.zeros((len(infos), len(infos[0]), self.envs.env_ref.env.config['total_action_dim']))
                
                for i in range(len(infos)):
                    for j in range(len(infos[0])):
                        masked[i,j,:] = np.array(infos[i][j]['available_actions'])
            else:
                masked = None

            self.buffer[agent_id].insert(share_obs,
                                         np.array(list(obs[:, agent_id])),
                                         rnn_states[:, agent_id],
                                         rnn_states_critic[:, agent_id],
                                         actions[:, agent_id],
                                         action_log_probs[:, agent_id],
                                         values[:, agent_id],
                                         rewards[:, agent_id],
                                         masks[:, agent_id],
                                         available_actions = masked[:, agent_id])

    def update_rewards(self, new_reward):
        for agent_id in range(self.num_agents):
            
            self.buffer[agent_id].update_rewards(new_reward[agent_id])

    @torch.no_grad()
    def eval(self, total_num_steps):
        eval_episode_rewards = []
        eval_obs = self.eval_envs.reset()

        eval_rnn_states = np.zeros((self.n_eval_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size),
                                   dtype=np.float32)
        eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)

        for eval_step in range(self.episode_length):
            eval_temp_actions_env = []
            for agent_id in range(self.num_agents):
                self.trainer[agent_id].prep_rollout()
                eval_action, eval_rnn_state = self.trainer[agent_id].policy.act(np.array(list(eval_obs[:, agent_id])),
                                                                                eval_rnn_states[:, agent_id],
                                                                                eval_masks[:, agent_id],
                                                                                deterministic=True)

                eval_action = eval_action.detach().cpu().numpy()
                
                if self.eval_envs.action_space[agent_id].__class__.__name__ == 'MultiDiscrete':
                    for i in range(self.eval_envs.action_space[agent_id].shape):
                        eval_uc_action_env = np.eye(self.eval_envs.action_space[agent_id].high[i] + 1)[
                            eval_action[:, i]]
                        if i == 0:
                            eval_action_env = eval_uc_action_env
                        else:
                            eval_action_env = np.concatenate((eval_action_env, eval_uc_action_env), axis=1)
                elif self.eval_envs.action_space[agent_id].__class__.__name__ == 'Discrete':
                    eval_action_env = np.squeeze(np.eye(self.eval_envs.action_space[agent_id].n)[eval_action], 1)
                else:
                    raise NotImplementedError

                eval_temp_actions_env.append(eval_action_env)
                eval_rnn_states[:, agent_id] = _t2n(eval_rnn_state)

            
            eval_actions_env = []
            for i in range(self.n_eval_rollout_threads):
                eval_one_hot_action_env = []
                for eval_temp_action_env in eval_temp_actions_env:
                    eval_one_hot_action_env.append(eval_temp_action_env[i])
                eval_actions_env.append(eval_one_hot_action_env)

            
            eval_obs, eval_rewards, eval_dones, eval_infos = self.eval_envs.step(eval_actions_env)
            eval_episode_rewards.append(eval_rewards)

            eval_rnn_states[eval_dones == True] = np.zeros(
                ((eval_dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
            eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
            eval_masks[eval_dones == True] = np.zeros(((eval_dones == True).sum(), 1), dtype=np.float32)

        eval_episode_rewards = np.array(eval_episode_rewards)

        eval_train_infos = []
        for agent_id in range(self.num_agents):
            eval_average_episode_rewards = np.mean(np.sum(eval_episode_rewards[:, :, agent_id], axis=0))
            eval_train_infos.append({'eval_average_episode_rewards': eval_average_episode_rewards})
            #("eval average episode rewards of agent%i: " % agent_id + str(eval_average_episode_rewards))

        self.log_train(eval_train_infos, total_num_steps)

    @torch.no_grad()
    def render(self):
        all_frames = []
        for episode in range(self.all_args.render_episodes):
            episode_rewards = []
            obs = self.envs.reset()
            if self.all_args.save_gifs:
                image = self.envs.render('rgb_array')[0][0]
                all_frames.append(image)

            rnn_states = np.zeros((self.n_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size),
                                  dtype=np.float32)
            masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)

            for step in range(self.episode_length):
                calc_start = time.time()

                temp_actions_env = []
                for agent_id in range(self.num_agents):
                    if not self.use_centralized_V:
                        share_obs = np.array(list(obs[:, agent_id]))
                    self.trainer[agent_id].prep_rollout()
                    action, rnn_state = self.trainer[agent_id].policy.act(np.array(list(obs[:, agent_id])),
                                                                          rnn_states[:, agent_id],
                                                                          masks[:, agent_id],
                                                                          deterministic=True)

                    action = action.detach().cpu().numpy()
                    
                    if self.envs.action_space[agent_id].__class__.__name__ == 'MultiDiscrete':
                        for i in range(self.envs.action_space[agent_id].shape):
                            uc_action_env = np.eye(self.envs.action_space[agent_id].high[i] + 1)[action[:, i]]
                            if i == 0:
                                action_env = uc_action_env
                            else:
                                action_env = np.concatenate((action_env, uc_action_env), axis=1)
                    elif self.envs.action_space[agent_id].__class__.__name__ == 'Discrete':
                        action_env = np.squeeze(np.eye(self.envs.action_space[agent_id].n)[action], 1)
                    else:
                        raise NotImplementedError

                    temp_actions_env.append(action_env)
                    rnn_states[:, agent_id] = _t2n(rnn_state)

                
                actions_env = []
                for i in range(self.n_rollout_threads):
                    one_hot_action_env = []
                    for temp_action_env in temp_actions_env:
                        one_hot_action_env.append(temp_action_env[i])
                    actions_env.append(one_hot_action_env)

                
                obs, rewards, dones, infos = self.envs.step(actions_env)
                episode_rewards.append(rewards)

                rnn_states[dones == True] = np.zeros(((dones == True).sum(), self.recurrent_N, self.hidden_size),
                                                     dtype=np.float32)
                masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
                masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)

                if self.all_args.save_gifs:
                    image = self.envs.render('rgb_array')[0][0]
                    all_frames.append(image)
                    calc_end = time.time()
                    elapsed = calc_end - calc_start
                    if elapsed < self.all_args.ifi:
                        time.sleep(self.all_args.ifi - elapsed)

            episode_rewards = np.array(episode_rewards)
            for agent_id in range(self.num_agents):
                average_episode_rewards = np.mean(np.sum(episode_rewards[:, :, agent_id], axis=0))
                #print("eval average episode rewards of agent%i: " % agent_id + str(average_episode_rewards))

        if self.all_args.save_gifs:
            imageio.mimsave(str(self.gif_dir) + '/render.gif', all_frames, duration=self.all_args.ifi)