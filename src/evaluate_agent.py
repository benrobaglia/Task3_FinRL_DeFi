import numpy as np
import pandas as pd
import pickle
import os
import argparse
import ast
from stable_baselines3 import PPO
from custom_env_folder.custom_env2 import Uniswapv3Env

def load_best_config_from_study(expname, window_num):
    """Load the best configuration from study results CSV files."""
    config = {}
    
    # Load phase 0 study results (contains action_values and other initial params)
    study0_path = f"output/{expname}/study_result_rolling_{window_num}_0.csv"
    if not os.path.exists(study0_path):
        study0_path = f"src/output/{expname}/study_result_rolling_{window_num}_0.csv"
    
    if os.path.exists(study0_path):
        df0 = pd.read_csv(study0_path)
        # Find the best trial (highest value)
        best_trial0 = df0.loc[df0['value'].idxmax()]
        
        # Extract action_values (it's stored as a string, need to parse it)
        if 'params_action_values' in best_trial0:
            action_values_str = best_trial0['params_action_values']
            config['action_values'] = ast.literal_eval(action_values_str)
        
        print(f"Loaded from phase 0 study: action_values={config.get('action_values')}")
    
    # Load phase 1 study results (may contain other optimized params)
    study1_path = f"output/{expname}/study_result_rolling_{window_num}_1.csv"
    if not os.path.exists(study1_path):
        study1_path = f"src/output/{expname}/study_result_rolling_{window_num}_1.csv"
    
    if os.path.exists(study1_path):
        df1 = pd.read_csv(study1_path)
        # Find the best trial (highest value)
        best_trial1 = df1.loc[df1['value'].idxmax()]
        print(f"Best trial from phase 1: value={best_trial1['value']:.2f}")
    
    # Set default values for parameters not in study results
    config.setdefault('delta', 0.05)
    config.setdefault('action_values', [0, 30, 60, 90])  # Default if not found
    config.setdefault('x', 10)
    config.setdefault('gas_fee', 5)
    config['action_values'] = [0, 40, 60, 80]  # Default action values if not found

    return config

def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained agent on a given data file.")
    parser.add_argument("--data_file", type=str, required=True, help="Path to the data file (CSV) for evaluation.")
    parser.add_argument("--window", type=int, default=15, help="Window number to use (0-9, default: 9 for the last model)")
    parser.add_argument("--expname", type=str, default="20250807_PPOUniswap_CVRFE", help="Experiment name (default: 20250807_PPOUniswap_CVRFE)")
    args = parser.parse_args()
    
    # Load config from study results
    print(f"Loading best configuration from study results...")
    config = load_best_config_from_study(args.expname, args.window)
    print(f"Configuration loaded: delta={config['delta']}, action_values={config['action_values']}, x={config['x']}, gas_fee={config['gas_fee']}")
    
    # Use specified window or default to last model (window 9)
    window_num = args.window
    expname = args.expname
    
    print(f"Using experiment: {expname}, Window: {window_num}")
    
    # Check if model files exist
    vol_model_path = f"src/output/{expname}/rf_models/vol_rfe_model_{window_num}.pkl"
    features_info_path = f"src/output/{expname}/rf_models/features_info_{window_num}.pkl"
    model_path = f"src/output/{expname}/PPOpolicy_{window_num}.zip"
    
    # Check paths from current directory
    if not os.path.exists(vol_model_path):
        # Try without src/ prefix
        vol_model_path = f"output/{expname}/rf_models/vol_rfe_model_{window_num}.pkl"
        features_info_path = f"output/{expname}/rf_models/features_info_{window_num}.pkl"
        model_path = f"output/{expname}/PPOpolicy_{window_num}.zip"
    
    # Verify files exist
    if not os.path.exists(vol_model_path):
        raise FileNotFoundError(f"RF model not found at: {vol_model_path}")
    if not os.path.exists(features_info_path):
        raise FileNotFoundError(f"Features info not found at: {features_info_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"PPO model not found at: {model_path}")
    
    print(f"Found model files for experiment {expname}, window {window_num}")
    
    # 2. Load the data
    print("2. Loading test data...")
    uni_test = pd.read_csv(args.data_file)
    print(f"Test data shape: {uni_test.shape}")
    
    # 3. Load the features from the cvrfe and the rf model
    print("3. Loading RF model and features info...")
    with open(vol_model_path, 'rb') as f:
        vol_rfe = pickle.load(f)
    print("RF model loaded")
    
    # Load the features info to get the correct combined features
    with open(features_info_path, 'rb') as f:
        features_info = pickle.load(f)
    print("Features info loaded")
    print(f"Features for volatility: {features_info.get('features_vol', [])}")
    
    # 4. Initialize the test env with process_features=True
    print("4. Processing features...")
    tenv = Uniswapv3Env(
        delta=config['delta'], 
        action_values=config['action_values'], 
        market_data=uni_test, 
        market_features=None,
        x=config['x'], 
        gas=config['gas_fee'],
        process_features=True
    )
    uni_test_process = tenv.market_data.copy()
    print(f"Processed data shape: {uni_test_process.shape}")
    
    # 5. Compute vol_pred with the rf model on the combined features
    
    print("5. Computing vol_pred...")
    features = list(uni_test_process.columns)
    to_remove = ['timestamp', 'price', 'open_price', 'high_price', 'low_price', 'closed_price', 'log_returns']
    for col in to_remove:
        if col in features:
            features.remove(col)

    feature_data = uni_test_process[features].copy()
    # Clean data
    feature_data = feature_data.replace([np.inf, -np.inf], np.nan)
    feature_data = feature_data.fillna(feature_data.median())
    feature_data = feature_data.clip(-1e10, 1e10)
        
    uni_test_process['vol_pred'] = vol_rfe.estimator_.predict(vol_rfe.transform(feature_data))
    print("vol_pred computed")
    
    # 6. Init test env with process_features=False and combined_features
    print("6. Creating final test environment...")
    
    # Use the combined features from the saved features_info
    combined_features = features_info.get('combined_features', [])
    
    # Ensure vol_pred is in the combined features if not already present
    if 'vol_pred' not in combined_features:
        print("Warning: vol_pred not found in combined_features from features_info, adding it")
        combined_features.append('vol_pred')
    
    print(f"Combined features from features_info: {combined_features}")
    print(f"Number of combined features: {len(combined_features)}")

    test_env = Uniswapv3Env(
        delta=config['delta'], 
        action_values=config['action_values'], 
        market_data=uni_test_process, 
        market_features=combined_features,
        x=config['x'], 
        gas=config['gas_fee'],
        process_features=False
    )
    print(f"Test environment created with observation space: {test_env.observation_space}")
    print(f"Environment features list: {test_env.features_list}")
    print(f"Total observation length: {len(test_env.features_list)}")
    
    # 7. Load the policy
    print("7. Loading policy...")
    print(f"Loading model from: {model_path}")
    rlmodel = PPO.load(model_path)
    print("Policy loaded successfully")
    
    # 8. Run the policy on the environment on 1 episode
    print("8. Running evaluation...")
    obs, _ = test_env.reset()
    print(f"Initial observation shape: {obs.shape}")
    
    total_reward = 0
    step_count = 0
    
    for step in range(len(uni_test_process)):
        action, _states = rlmodel.predict(obs, deterministic=True)
        obs, rewards, done, truncated, info = test_env.step(action)
        total_reward += rewards
        step_count += 1
        
        if step % 100 == 0:
            print(f"Step {step}: Action={test_env.action_values[action]}, Reward={rewards:.4f}, Total={total_reward:.4f}")
        
        if done or truncated:
            break
    
    print(f"\n=== EVALUATION COMPLETE ===")
    print(f"Experiment: {expname}")
    print(f"Window: {window_num}")
    print(f"Total Steps: {step_count}")
    print(f"Total Reward: {total_reward:.4f}")
    print(f"Average Reward per Step: {total_reward/step_count:.4f}")
    
    # Save results
    test_history = test_env.history
    results_df = pd.DataFrame(test_history)
    output_file = f"evaluation_results_window_{window_num}.csv"
    results_df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    main()
