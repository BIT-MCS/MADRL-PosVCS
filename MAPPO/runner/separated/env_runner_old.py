"""
# @Time    : 2021/7/1 7:14 下午
# @Author  : hezhiqiang01
# @Email   : hezhiqiang01@baidu.com
# @File    : env_runner.py
"""

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


def _t2n(x):
    return x.detach().cpu().numpy()


class EnvRunner(Runner):
    def __init__(self, config):
        super(EnvRunner, self).__init__(config)

    def run(self):
        self.warmup()

        start = time.time()
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads

        # 启用路径储存，储存在同一个run下
        rend = False
        if self.all_args.generate_outputs:
            ret_dict = {'observation':[],'rewards':[],'dones': [],'info':[]}
            episodes = 1 #只evaluate一个回合
            for i in range(len(self.envs.envs)):
                # 加入噪音
                self.envs.envs[i].env._data_amount_true_env()
                # 更改收集能力为预定收集能力
                self.envs.envs[i].env.config['maximum_collect'] = self.envs.envs[i].env.config['render_collect']
                # 更改环境的转移概率为实测,先判断实测为1即没转移完是不可以走的, 这个在step里实现
                if self.envs.envs[i].env.config['transition_mode'] != 'full_success':
                    self.envs.envs[i].env.config['transition_mode'] = 'full_success'
                    print('Force env following rendering scheme, check env_runner.py for more info')
            rend = True

        # if self.envs.envs[0].env.config['use_dual_descent']:
        #     #agent_num = self.envs.envs[0].env.config['agent_num']
        #     legal_pos = self.envs.envs[0].env.legal_pos
        #     poi_pos = self.envs.envs[0].env.poi
        #     floor_list = self.envs.envs[0].env.config['floor_list']
        #     #{env1:{fl:{pos1:[], pos2,...},...},...}
        #     deltasum_dict = {i:{fl:{pos:0 for pos in legal_pos[fl]} for fl in floor_list} \
        #         for i in range(len(self.envs.envs))}
            
        # if self.envs.env_ref.config['use_dual_descent']:
        #     #agent_num = self.envs.envs[0].env.config['agent_num']
        #     legal_pos = self.envs.env_ref.legal_pos
        #     poi_pos = self.envs.env_ref.poi
        #     floor_list = self.envs.env_ref.config['floor_list']
        #     #{env1:{fl:{pos1:[], pos2,...},...},...}
        #     deltasum_dict = {i:{fl:{pos:0 for pos in legal_pos[fl]} for fl in floor_list} \
        #         for i in range(len(self.envs.nenvs))}
            

        for episode in range(episodes):
            if self.use_linear_lr_decay:
                for agent_id in range(self.num_agents):
                    self.trainer[agent_id].policy.lr_decay(episode, episodes)

            for step in range(self.episode_length):
                # Sample actions
                values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env = self.collect(step)

                # Obser reward and next obs
                obs, rewards, dones, infos = self.envs.step(actions_env, rend)

                # 启用路径储存，储存在同一个run下
                if self.all_args.generate_outputs:
                    ret_dict['observation'].append(obs[0])
                    ret_dict['rewards'].append(rewards[0])
                    ret_dict['dones'].append(dones[0])
                    ret_dict['info'].append(infos[0]) 

                data = obs, rewards, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic

                # if self.envs.envs[0].env.config['use_dual_descent'] and not rend:
                #     for i in range(len(self.envs.envs)):
                #         curr_env = self.envs.envs[i]
                #         for uav in range(curr_env.env.config['agent_num']):
                #             pos = infos[i][uav]['aoi_reduce'][0]
                #             delta_amount =infos[i][uav]['aoi_reduce'][1]
                #             deltasum_dict[i][pos[0]][pos[1]] += delta_amount

                        # if step == self.episode_length - 1:
                        #     # 最后一步需要把那些没去过penalty直接给上
                        #     #print('hi')
                        #     for fl in floor_list:
                        #         for pos in poi_pos[fl]:
                        #             if deltasum_dict[i][fl][pos] == 0:
                        #                 # poi位置没去过
                        #                 deltasum_dict[i][fl][pos] += self.episode_length
                

                # insert data into buffer
                self.insert(data)

            # compute return and update network
            self.compute()

            # 如果使用dual descent，要在这里要对lambda提升
            # 在环境里写lambda 然后单独拿出来提升
            # 提升的时候需要一整个episode的信息，需要喂给环境
            # if self.envs.envs[0].env.config['use_dual_descent'] and not rend:
            #     # 对每个环境里的lambda进行更新
            #     for i in range(len(self.envs.envs)):
            #         curr_env = self.envs.envs[i].env
            #         curr_env._compute_lambda_updates(deltasum_dict[i])

            train_infos = self.train()

            # post process
            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads

            # save model
            if (episode % self.save_interval == 0 or episode == episodes - 1):
                self.save()

            # log information
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

                #if self.env_name == "MPE":
                for agent_id in range(self.num_agents):
                    # idv_rews = []
                    # for info in infos:
                    #     if 'individual_reward' in info[agent_id].keys():
                    #         idv_rews.append(info[agent_id]['individual_reward'])
                    # train_infos[agent_id].update({'individual_rewards': np.mean(idv_rews)})
                    train_infos[agent_id].update(
                        {"average_episode_rewards": np.mean(self.buffer[agent_id].rewards) * self.episode_length})
                    print("episode rewards for agent {} is {}".format(agent_id, train_infos[agent_id]["average_episode_rewards"]))
                
                self.log_train(train_infos, total_num_steps)
            
                # # 环境中的信息总结
                # if episode % (10 * self.log_interval):
                #     self.envs.envs[0].env._runtime_summary()

            # eval
            if episode % self.eval_interval == 0 and self.use_eval:
                self.eval(total_num_steps)

        if self.all_args.generate_outputs:
        # 储存数据到指定路径
            try:
                os.remove('./generate_dict.pickle')
            except OSError as e:
                pass

            with open('generate_dict.pickle','wb') as f:
                pickle.dump(ret_dict, f)

    def warmup(self):
        # reset env
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
            
            # [agents, envs, dim]
            values.append(_t2n(value))
            action = _t2n(action)
            # rearrange action
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
                # TODO 这里改造成自己环境需要的形式即可
                action_env = actions
                # raise NotImplementedError

            actions.append(action)
            temp_actions_env.append(action_env)
            action_log_probs.append(_t2n(action_log_prob))
            rnn_states.append(_t2n(rnn_state))
            rnn_states_critic.append(_t2n(rnn_state_critic))

        # [envs, agents, dim]
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
            

            masked = np.zeros((infos.shape[0], infos.shape[1], 28))
            # TODO：没准有点慢，考虑vectorize
            for i in range(infos.shape[0]):
                for j in range(infos.shape[1]):
                    masked[i,j,:] = np.array(infos[i][j]['available_actions'])


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
                # rearrange action
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

            # [envs, agents, dim]
            eval_actions_env = []
            for i in range(self.n_eval_rollout_threads):
                eval_one_hot_action_env = []
                for eval_temp_action_env in eval_temp_actions_env:
                    eval_one_hot_action_env.append(eval_temp_action_env[i])
                eval_actions_env.append(eval_one_hot_action_env)

            # Obser reward and next obs
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
            print("eval average episode rewards of agent%i: " % agent_id + str(eval_average_episode_rewards))

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
                    # rearrange action
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

                # [envs, agents, dim]
                actions_env = []
                for i in range(self.n_rollout_threads):
                    one_hot_action_env = []
                    for temp_action_env in temp_actions_env:
                        one_hot_action_env.append(temp_action_env[i])
                    actions_env.append(one_hot_action_env)

                # Obser reward and next obs
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
                print("eval average episode rewards of agent%i: " % agent_id + str(average_episode_rewards))

        if self.all_args.save_gifs:
            imageio.mimsave(str(self.gif_dir) + '/render.gif', all_frames, duration=self.all_args.ifi)