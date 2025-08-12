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
import pickle
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.feature_selection import RFE, RFECV


import seaborn as sns
import matplotlib.pyplot as plt
sns.set_theme(style='white')
warnings.filterwarnings('ignore')
from stable_baselines3 import PPO
# from custom_env_folder.custom_env_ewma import Uniswapv3Env, CustomMLPFeatureExtractor
from custom_env_folder.custom_env2 import Uniswapv3Env, CustomMLPFeatureExtractor


'''
This code is used to build and test the reinforcement learning algorithm
on the imported custom environment for Uniswap V3
'''

def perform_rfe_rf(train_df, features, target_column, n_features_to_select=10):
    """
    Perform RFE for feature selection using Random Forest Regressor.
    
    Args:
        train_df (pd.DataFrame): Training dataframe with all features and target.
        features (list): List of feature column names.
        target_column (str): Column name of the target (e.g., 'future_price' or 'future_vol').
        n_features_to_select (int): Number of features to select.
    
    Returns:
        selected_features (list): List of selected feature names.
        rfe_model (RFE): The fitted RFE model (with underlying RF).
    """
    # Drop rows with NaN in features or target
    df_clean = train_df.dropna(subset=features + [target_column])
    
    X = df_clean[features]
    y = df_clean[target_column]
    
    # Random Forest as estimator
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    
    # RFE setup: eliminate step=1 feature per iteration
    rfe = RFE(estimator=rf, n_features_to_select=n_features_to_select, step=1)
    rfe.fit(X, y)
    
    # Get selected features
    selected_features = [f for f, s in zip(features, rfe.support_) if s]
    
    # print(f"Selected {len(selected_features)} features for {target_column}: {selected_features}")
    
    # Feature importance from RFE
    importances_full = np.full(len(features), np.nan)
    importances_full[rfe.support_] = rfe.estimator_.feature_importances_

    feature_importance_df = pd.DataFrame({
        'feature': features,
        'selected': rfe.support_,
        'ranking': rfe.ranking_,
        'importance': importances_full
    })
    
    return selected_features, feature_importance_df, rfe

def perform_cvrfe_rf(train_df, features, target_column, min_features_to_select=5):
    """
    Perform CV-RFE for automatic feature selection using Random Forest Regressor.

    Args:
    train_df (pd.DataFrame): Training dataframe with all features and target.
    features (list): List of feature column names.
    target_column (str): Column name of the target (e.g., 'future_price' or 'future_vol').

    Returns:
    selected_features (list): List of selected feature names.
    feature_importance_df (pd.DataFrame): DataFrame with feature importance scores from the fitted model.
    cvrfe_model (RFECV): The fitted RFECV model.
    """
    # Drop rows with NaN in features or target
    df_clean = train_df.dropna(subset=features + [target_column])
    X = df_clean[features]
    y = df_clean[target_column]
    
    # Data validation and cleaning
    print(f"Data shape before cleaning: {X.shape}")
    
    # Check for infinite values
    inf_mask = np.isinf(X.values)
    if inf_mask.any():
        inf_count = inf_mask.sum()
        print(f"Warning: Found {inf_count} infinite values in features. Replacing with NaN and then dropping.")
        X = X.replace([np.inf, -np.inf], np.nan)
        
    # Check for extremely large values (beyond float32 range)
    large_mask = np.abs(X.values) > np.finfo(np.float32).max
    if large_mask.any():
        large_count = large_mask.sum()
        print(f"Warning: Found {large_count} extremely large values. Replacing with NaN and then dropping.")
        X = X.mask(np.abs(X) > np.finfo(np.float32).max)
    
    # Remove rows with any remaining NaN values after cleaning
    before_drop = len(X)
    mask = X.notna().all(axis=1) & y.notna()
    X = X[mask]
    y = y[mask]
    after_drop = len(X)
    
    if before_drop != after_drop:
        print(f"Dropped {before_drop - after_drop} rows due to problematic values. Remaining: {after_drop}")
    
    # Check if we have enough data left
    if len(X) < 50:  # Minimum threshold for reliable CVRFE
        raise ValueError(f"Insufficient clean data for CVRFE: only {len(X)} samples remaining")
    
    print(f"Final data shape for CVRFE: {X.shape}")
    
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)

    # CV-RFE setup: uses cross-validation to select optimal number of features
    cvrfe = RFECV(estimator=rf, step=1, cv=5, scoring='neg_mean_squared_error', min_features_to_select=min_features_to_select, n_jobs=-1)
    cvrfe.fit(X, y)

    # Get selected features
    selected_features = [f for f, s in zip(features, cvrfe.support_) if s]
    print(f"Selected {len(selected_features)} features for {target_column}: {selected_features}")

    # Feature importance from CV-RFE
    importances_full = np.full(len(features), np.nan)
    importances_full[cvrfe.support_] = cvrfe.estimator_.feature_importances_

    feature_importance_df = pd.DataFrame({
        'feature': features,
        'selected': cvrfe.support_,
        'ranking': cvrfe.ranking_,
        'importance': importances_full
    })

    return selected_features, feature_importance_df, cvrfe



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


def optimize_ppo(trial, param_name, uni_train, uni_test, combined_features, model_name=None):
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
                           market_features=combined_features,
                           x=params['x'], 
                           gas=params['gas_fee'],
                           process_features=False
                           )
    
    test_env = Uniswapv3Env(delta=params['delta'], 
                        action_values=params['action_values'], 
                        market_data=uni_test, 
                        market_features=combined_features,
                        x=params['x'], 
                        gas=params['gas_fee'],
                        process_features=False
                        )
    uni_env.reset()
    
    # Define policy_kwargs with the custom feature extractor
    policy_kwargs = dict(
        features_extractor_class=CustomMLPFeatureExtractor,
        features_extractor_kwargs=dict(features_dim=128,
                                       activation=params['activation'],
                                       hidden_dims=params['dim_hidden_layers'],
                                       dropout_rate=0.1
                                       ),  # This dimension will be the output of the feature extractor
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
        if model_name is None:
            model_name = f"output/{expname}/PPOpolicy.zip"
        else:
            model_name = f"{model_name}.zip"
            
        rl_model.save(model_name)
        
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
    with open(os.path.join(config_dir, "uniswap_rl_param_2707.yaml"), "r") as f:
        params = yaml.safe_load(f)
    
    # import market data
    uni_table = pd.read_csv(params['filename'])

    uni_time = uni_table[['timestamp']]
    # uni_data = uni_table[['price']]
    
    split_size = 1500
    dfs_list = [uni_table.iloc[i:i + split_size].reset_index(drop=True) for i in range(0, len(uni_table), split_size)]
    dfs_list.pop()      # drop the last df which has different length
    
    times_list = [uni_time.iloc[i:i + split_size].reset_index(drop=True) for i in range(0, len(uni_time), split_size)]
    times_list.pop()    # drop the last df which has different length

    def optuna_study(param_filename, new_param_filename, study_result_filename, n_trials, uni_train, uni_test, combined_features, model_name=None):
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
                                    uni_test=uni_test,
                                    combined_features=combined_features,
                                    model_name=model_name
                                    )
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
        if new_param_filename == "uniswap_rl_param_2707_r1.yaml":
            existing_params['grid'] = ['ent_coef', 'gamma', 'clip_range']
        else:
            existing_params['grid'] = ['action_values', 'learning_rate', 'dim_hidden_layers', 'activation']
            
        yaml_file = os.path.join(config_dir, new_param_filename)
        with open(yaml_file, 'w') as file:
            yaml.dump(existing_params, file, default_flow_style=None)
        
        print("Update from ", param_filename, " to ", new_param_filename)
        return trial
    
    expname = "20250807_PPOUniswap_CVRFE" 
    
    # create a for loop to iterate over the data
    rolling_window = 5  # Define rolling window size
    i = len(dfs_list)
    n_trials = 20
    print(f"\nROLLING WINDOW: {i+1}/{len(dfs_list)-rolling_window}...")
    df0 = dfs_list[i-1]
    df1 = dfs_list[i-2]
    df2 = dfs_list[i-3]
    df3 = dfs_list[i-4]
    df4 = dfs_list[i-5]

    uni_train = pd.concat([df0, df1, df2, df3, df4], ignore_index=True)
    # uni_test = dfs_list[i+5]
    uni_test = df0

    time_train = pd.concat([times_list[i-1], times_list[i-2], times_list[i-3], times_list[i-4], times_list[i-5]], ignore_index=True)
    # time_test = times_list[i+5]
    
    '''
    study 0: Perform feature selection using RFE with Random Forest
    '''
    
    tenv = Uniswapv3Env(delta=params['delta'], 
                        action_values=params['action_values'], 
                        market_data=uni_train, 
                        market_features=None,  # Set to None to use default features
                        x=params['x'], 
                        gas=params['gas_fee'],
                        process_features=True
                        )
    
    uni_train_process = tenv.market_data.copy()
    
    features = list(uni_train_process.columns)
    to_remove = ['timestamp', 'price', 'open_price', 'high_price', 'low_price', 'closed_price', 'log_returns']
    for col in to_remove:
        if col in features:
            features.remove(col)
    
    # features = ['ma168', 'ema_5', 'tema_5', 'tema_10', 'dema_5', 'dema_10', 'roc_25', 'stddev_20', 'price_momentum_1', 'gap', 'macd']

    # uni_train_process['future_ret'] = uni_train_process['log_returns'].shift(-1)
    uni_train_process['future_vol'] = uni_train_process['ew_sigma'].shift(-1)
    uni_train_process = uni_train_process.dropna()  # Clean NaNs
    
    print("\nFeatures selection with CVRFE...")
    # features_vol, vol_feature_importance, vol_rfe = perform_rfe_rf(uni_train_process, features, 'future_vol', n_features_to_select=10)
    
    features_vol, vol_feature_importance, vol_rfe = perform_cvrfe_rf(uni_train_process, features, 'future_vol', min_features_to_select=3)
    
    os.makedirs(f"data/{expname}", exist_ok=True)
    vol_feature_importance.to_csv(f"data/{expname}/vol_feature_importance_{i}.csv", index=False)
    
    # Save the Random Forest models for later use in evaluation
    os.makedirs(f"output/{expname}/rf_models", exist_ok=True)
    vol_rfe_path = f"output/{expname}/rf_models/vol_rfe_model_{i}.pkl"
    with open(vol_rfe_path, 'wb') as f:
        pickle.dump(vol_rfe, f)
    print(f"Saved volatility RF model to: {vol_rfe_path}")
    
    # Save the feature list and combined features for reference
    features_info = {
        'features_vol': features_vol,
        'combined_features': None,  # Will be updated below
        'market_features_base': ['ma24', 'ma168', 'bb_width_20', 'adxr', 'price_momentum_24', 'aroon_osc_25', 'cp_prob'],
        'all_features': features
    }

    market_features_base = ['ma24', 'ma168', 'bb_width_20', 'adxr', 'price_momentum_24', 'aroon_osc_25', 'cp_prob']

    combined_features = list(set(market_features_base) | set(features_vol))
    
    # Add the prediction columns to the combined features
    # uni_train_process['price_pred'] = price_rfe.estimator_.predict(price_rfe.transform(uni_train_process[features]))
    uni_train_process['vol_pred'] = vol_rfe.estimator_.predict(vol_rfe.transform(uni_train_process[features]))
    
    if 'ew_sigma' in combined_features:
        combined_features.remove('ew_sigma')
    
    # Add the prediction columns to the combined features
    # combined_features.append('price_pred')
    combined_features.append('vol_pred')
    
    # Update and save the complete features information
    features_info['combined_features'] = combined_features
    features_info_path = f"output/{expname}/rf_models/features_info_{i}.pkl"
    with open(features_info_path, 'wb') as f:
        pickle.dump(features_info, f)
    print(f"Saved features info to: {features_info_path}")

    # print("Combined features selected for both future price and volatility:", combined_features)
    tenv = Uniswapv3Env(delta=params['delta'], 
                        action_values=params['action_values'], 
                        market_data=uni_test, 
                        market_features=None,
                        x=params['x'], 
                        gas=params['gas_fee'],
                        process_features=True
                        )
    
    uni_test_process = tenv.market_data.copy()
    # uni_test_process['price_pred'] = price_rfe.estimator_.predict(price_rfe.transform(uni_test_process[features]))
    uni_test_process['vol_pred'] = vol_rfe.estimator_.predict(vol_rfe.transform(uni_test_process[features]))

    '''
    study 1: optimize action_value, learning_rate, dim_hidden_layers, activation
    '''
    best_performance = -float('inf')
    best_trial_num = -1
    trial_counter = 0
    out_filename = "output/{}/study_result_rolling_{}_0.csv".format(expname,i)
    trial1 = optuna_study("uniswap_rl_param_2707.yaml",
                            "uniswap_rl_param_2707_r1.yaml",
                            out_filename, n_trials, uni_train_process, uni_test_process, combined_features)
        
    '''
    study 2: optimize ent_coef, gamma, clip_range
    '''
    best_performance = -float('inf')
    best_trial_num = -1
    trial_counter = 0
    out_filename = "output/{}/study_result_rolling_{}_1.csv".format(expname,i)
    model_name = f"output/{expname}/PPOpolicy_{i}"
    trial2 = optuna_study("uniswap_rl_param_2707_r1.yaml",
                            "uniswap_rl_param_2707.yaml",
                            out_filename, n_trials, uni_train_process, uni_test_process, combined_features, model_name)
    
    # # Rollback if the best trial is not the first one
    # if trial1.value > trial2.value:
    #     print("Rollback to the first study...")
    #     model_name = f"output/{expname}/PPOpolicy1"
    # else:
    #     print("Rollback to the second study...")
    #     model_name = f"output/{expname}/PPOpolicy2"
    
    
    '''
    study 3: after optimization, visualize the test data and save plots to folder
    '''
    # load new hyperparam file
    with open(os.path.join(config_dir, "uniswap_rl_param_2707_r1.yaml"), "r") as f:
        params = yaml.safe_load(f)
        
    # create test environment
    test_env = Uniswapv3Env(delta=params['delta'], 
                    action_values=params['action_values'], 
                    market_data=uni_test_process, 
                    market_features=combined_features,
                    x=params['x'], 
                    gas=params['gas_fee'],
                    process_features=False
                    )
    
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
