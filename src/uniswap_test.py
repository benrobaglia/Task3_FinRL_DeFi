'''
Import packages
'''
import numpy as np
import pandas as pd
import os
import sys
import functools
from functools import partial
import torch
import torch.nn as nn
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
import yaml
import optuna
import warnings

import seaborn as sns
import matplotlib.pyplot as plt
sns.set_theme(style='white')
warnings.filterwarnings('ignore')
from stable_baselines3 import PPO
# from custom_env_folder.custom_env_ewma import Uniswapv3Env, CustomMLPFeatureExtractor
from custom_env_folder.custom_env import Uniswapv3Env, CustomMLPFeatureExtractor
'''
This code is used to build and test the reinforcement learning algorithm
on the imported custom environment for Uniswap V3
'''
class RewardLoggerCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(RewardLoggerCallback, self).__init__(verbose)
        self.episode_rewards = []
        self.current_episode_reward = 0.0

    def _on_step(self) -> bool:
        self.current_episode_reward += self.locals['rewards'][0]
        if self.locals['dones'][0]:  # Checking for the end of an episode
            self.episode_rewards.append(self.current_episode_reward)
            self.current_episode_reward = 0.0
        return True

    def _on_training_end(self) -> None:
        np.save("rewards.npy", self.episode_rewards)  # Save rewards to file for later use
        
         
def evaluate_model(model, eval_env):
    """
    Evaluate a trained model on a specified environment for a certain number of episodes.
    Parameters:
        model: The trained model.
        eval_env: The environment to evaluate the model on.
        num_episodes (int): The number of episodes to run the model.
    Returns:
        mean_reward (float): The mean reward achieved by the model over the specified number of episodes.
    """

    obs,_ = eval_env.reset()
    episode_rewards = 0
    done = False
    truncated = False
    while not (done or truncated):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = eval_env.step(action)
        episode_rewards += reward
        
    return episode_rewards


def optimize_ppo(trial, param_name, uni_train, uni_test):
    '''
    Use optuna to optimize the hyperparameters
    Parameters:
        param_name (str): YAML filename for parameters
        uni_train (pd.DataFrame): training data
        uni_test (pd.DataFrame): testing data
    Returns:
        mean_reward (float): The mean reward achieved by the model over the specified number of episodes.
    '''
    
    global trial_counter
    trial_counter += 1
    print("Trial Count: ", trial_counter)
    
    base_dir = os.getcwd()
    config_dir = os.path.join(base_dir, "config")
    # Load hyperparam file
    with open(os.path.join(config_dir, param_name), "r") as f:
        params = yaml.safe_load(f)

    # Access the hyperparameters and grid from the config
    hyperparameters = params["hyperparameters"]
    grid = params["grid"]
    # Iterate over the hyperparameters and update p accordingly
    for param, (values, dtype) in hyperparameters.items():
        if param in grid:
            if dtype == "cat":
                params[param] = trial.suggest_categorical(param, values)
            elif dtype == "int":
                if len(values) == 3:
                    params[param] = trial.suggest_int(param, values[0], values[1], step=values[2])
                else:
                    params[param] = trial.suggest_int(param, values[0], values[1])
            elif dtype == "float":
                if len(values) == 3:
                    params[param] = trial.suggest_float(param, values[0], values[1], step=values[2])
                else:
                    params[param] = trial.suggest_float(param, values[0], values[1])
            else:
                print("Choose an available dtype!")
                sys.exit()
                
    
    # Fix randomness
    seed = params['seed']
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Instantiate the model
    uni_env = Uniswapv3Env(delta=params['delta'], 
                           action_values=params['action_values'], 
                           market_data=uni_train, 
                           x=params['x'], 
                           gas=params['gas_fee'])
    
    test_env = Uniswapv3Env(delta=params['delta'], 
                        action_values=params['action_values'], 
                        market_data=uni_test, 
                        x=params['x'], 
                        gas=params['gas_fee'])
    uni_env.reset()
    
    # Define policy_kwargs with the custom feature extractor
    policy_kwargs = dict(
        features_extractor_class=CustomMLPFeatureExtractor,
        features_extractor_kwargs=dict(features_dim=128,
                                       activation=params['activation'],
                                       hidden_dim=params['dim_hidden_layers']),  # This dimension will be the output of the feature extractor
    )

    # Create the PPO model with the custom MLP policy and feature extractor
    rl_model = PPO("MlpPolicy", uni_env, n_steps=len(uni_env.market_data) // 3, 
                    learning_rate=params['learning_rate'],
                    gamma=params['gamma'],
                    gae_lambda=params['gae_lambda'],  
                    clip_range=params['clip_range'],
                    ent_coef=params['ent_coef'],
                    vf_coef=params['vf_coef'],  
                    target_kl=params['target_kl'],                 
                    seed=seed,
                    policy_kwargs=policy_kwargs, 
                    batch_size=params['batch_size'],
                    verbose=0)
    # Train the model
    stop_train_callback = StopTrainingOnNoModelImprovement(max_no_improvement_evals=5, min_evals=5, verbose=0)
    eval_callback = EvalCallback(Monitor(uni_env), eval_freq=len(uni_env.market_data) // 3, callback_after_eval=stop_train_callback, verbose=0)
    reward_logger = RewardLoggerCallback()
    callback = CallbackList([eval_callback, reward_logger])
    
    rl_model.learn(total_timesteps=params['total_timesteps_1'],
                   progress_bar=False,
                   callback=callback)
    
    mean_reward = evaluate_model(rl_model, Monitor(test_env))
    
    # At the end of your optimize_ppo function, before returning mean_reward:
    current_performance = mean_reward
    
    global best_performance, best_trial_num, expname
    if current_performance > best_performance:
        best_performance = current_performance
        best_trial_num = trial.number
        
        # Save model and other relevant information
        model_path = "output/{}/PPOpolicy.zip".format(expname)
        rl_model.save(model_path)
        
    return mean_reward


####################################################################
if __name__ == "__main__":
    
    # create directory
    base_dir = os.getcwd()
    output_dir = os.path.join(base_dir, "output")
    config_dir = os.path.join(base_dir, "config")
    plot_dir = os.path.join(base_dir, "plot")
    data_dir = os.path.join(base_dir, "data")
    
    dirs = [output_dir, config_dir, plot_dir, data_dir]
    for dir in dirs:
        if not os.path.exists(dir):
            os.makedirs(dir)  
    
    # Load hyperparam file
    with open(os.path.join(config_dir, "uniswap_rl_param_1108.yaml"), "r") as f:
        params = yaml.safe_load(f)
    
    # import market data
    uni_table = pd.read_csv(params['filename'])

    uni_time = uni_table[['timestamp']]
    uni_data = uni_table[['price']]
    
    split_size = 1500
    dfs_list = [uni_data.iloc[i:i + split_size].reset_index(drop=True) for i in range(0, len(uni_data), split_size)]
    dfs_list.pop()      # drop the last df which has different length
    
    times_list = [uni_time.iloc[i:i + split_size].reset_index(drop=True) for i in range(0, len(uni_time), split_size)]
    times_list.pop()    # drop the last df which has different length

    def optuna_study(param_filename, new_param_filename, study_result_filename, n_trials, uni_train, uni_test):
        '''
        Use optuna to optimize the hyperparameter and save it to a csv file
        Parameters:
            param_filename (str): name of YAML file
            new_param_filename (str): name of the YAML file that we want to update its parameters
            study_result_filename (str): name of CSV file to save study result
            n_trials (int): number of trials
            uni_trains (pd.DataFrame): training data
            uni_test (pd.DataFrame): testing data
        Returns:
            trial (dictionary): optimized parameters for the best trial
        '''
        study = optuna.create_study(direction='maximize')
        partial_optimize_ppo = partial(optimize_ppo, 
                                    param_name=param_filename, 
                                    uni_train=uni_train,
                                    uni_test=uni_test)
        study.optimize(partial_optimize_ppo, 
                    n_trials=n_trials, 
                    # n_jobs=-1,
                    show_progress_bar=True)

        print("Best trial: ")
        trial = study.best_trial
        print(" Value: ", trial.value)
        print(" Params: ")
        for key, value in trial.params.items():
            print(f"{key}: {value}")
            
        # Saving study results
        study_df = study.trials_dataframe()
        study_df.to_csv(study_result_filename.format(expname))
        
        # update all parameters
        # read current YAML file
        yaml_file = os.path.join(config_dir, param_filename)
        with open(yaml_file, 'r') as file:
            existing_params = yaml.safe_load(file)
            
        # use update() to merge new parameters into the old dictionary
        if existing_params is None:
            existing_params = {}  # if YAML file is empty, initialize a new dictionary
        existing_params.update(trial.params)
        
        # keep the grid the same
        if new_param_filename == "uniswap_rl_param_1108_r1.yaml":
            existing_params['grid'] = ['ent_coef', 'gamma', 'clip_range']
        else:
            existing_params['grid'] = ['action_values', 'learning_rate', 'dim_hidden_layers', 'activation']
            
        # Save the current optimization into uniswap_rl_param_1108_r1.yaml
        yaml_file = os.path.join(config_dir, new_param_filename)
        with open(yaml_file, 'w') as file:
            yaml.dump(existing_params, file, default_flow_style=None)
        
        print("Update from ", param_filename, " to ", new_param_filename)
        return trial
    
    expname = "20241122_PPOUniswap" 
    
    # create a for loop to iterate over the data
    for i in range(len(dfs_list)-5):
        n_trials = 5
        print("ROLLING WINDOW:", i)
        df0 = dfs_list[i]
        df1 = dfs_list[i+1]
        df2 = dfs_list[i+2]
        df3 = dfs_list[i+3]
        df4 = dfs_list[i+4]
                
        uni_train = pd.concat([df0, df1, df2, df3, df4], ignore_index=True)
        uni_test = dfs_list[i+5]
        
        time_train = pd.concat([times_list[i], times_list[i+1], times_list[i+2], times_list[i+3], times_list[i+4]], ignore_index=True)
        time_test = times_list[i+5]
        '''
        study 1: optimize action_value, learning_rate, dim_hidden_layers, activation
        '''
        best_performance = -float('inf')
        best_trial_num = -1
        trial_counter = 0
        out_filename = "output/{}/study_result_rolling_{}_0.csv".format(expname,i)
        trial = optuna_study("uniswap_rl_param_1108.yaml",
                             "uniswap_rl_param_1108_r1.yaml",
                             out_filename, n_trials, uni_train, uni_test)
            
        '''
        study 2: optimize ent_coef, gamma, clip_range
        '''
        best_performance = -float('inf')
        best_trial_num = -1
        trial_counter = 0
        out_filename = "output/{}/study_result_rolling_{}_1.csv".format(expname,i)
        trial = optuna_study("uniswap_rl_param_1108_r1.yaml",
                             "uniswap_rl_param_1108.yaml",
                             out_filename, n_trials, uni_train, uni_test)
        
        '''
        study 3: after optimization, visualize the test data and save plots to folder
        '''
        # load new hyperparam file
        with open(os.path.join(config_dir, "uniswap_rl_param_1108_r1.yaml"), "r") as f:
            params = yaml.safe_load(f)
            
        # create test environment
        test_env = Uniswapv3Env(delta=params['delta'], 
                        action_values=params['action_values'], 
                        market_data=uni_test, 
                        x=params['x'], 
                        gas=params['gas_fee'])
        
        model_name = "output/{}/PPOpolicy".format(expname)
        rlmodel = PPO.load(model_name)
        obs, _ = test_env.reset()

        results = []
        # Run the test
        for step in range(len(uni_test)):
            action, _states = rlmodel.predict(obs, deterministic=True)
            obs, rewards, done, truncated, info = test_env.step(action)
            results.append((obs, action, rewards))

            if done:
                break
            
        # save the data of testing enviroment
        test_history = test_env.history
        df = pd.DataFrame(test_history)
        
        # save to a .csv table under /data/ folder
        output_dir = os.path.join(data_dir, expname)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir) 
        filename = "test_history_{}.csv".format(i)
        file_path = os.path.join(output_dir, filename)
            
        df.to_csv(file_path, index=False)
            
        def plot_cumulative(df, times):
            reward_history = df['Reward']
            cumum_reward = np.cumsum(reward_history)
            values = df['Value']
            plt.figure(figsize=(12, 6))

            plt.plot(times, cumum_reward + values - values[0], label='Net Asset Value Change', color='g')
            plt.plot(times, cumum_reward, label='Cumulative Reward', color='r')
            plt.plot(times, values, label='Asset Value', color='b')

            plt.xlabel('Timestep')
            plt.ylabel('Value')
            plt.legend()

            output_dir = os.path.join(plot_dir, expname)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir) 
            filename = "plt_rw_{}_cumulative.png".format(i)
            file_path = os.path.join(output_dir, filename)
            
            plt.savefig(file_path, format='png')
            
        def plot_price(df, times):
            p_history = df['Price']
            pl_history = df['Price_Lower']
            pu_history = df['Price_Upper']

            plt.figure(figsize=(12, 6))
            plt.plot(times, p_history, label='Price', color='g')
            plt.plot(times, pl_history, label='Price_Lower', color='b')
            plt.plot(times, pu_history, label='Price_Upper', color='r')

            plt.xlabel('Timestep')
            plt.ylabel('Price')
            plt.legend()
            
            output_dir = os.path.join(plot_dir, expname)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir) 
            filename = "plt_rw_{}_price.png".format(i)
            file_path = os.path.join(output_dir, filename)
            
            plt.savefig(file_path, format='png')
            
        def make_plots(df, times):
            times = times['timestamp']
            times = times[168:]
            times = times[:-1]
            times = pd.to_datetime(times)
            plot_cumulative(df, times)
            plot_price(df, times)

        make_plots(df, time_test)