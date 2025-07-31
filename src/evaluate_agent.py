import numpy as np
import pandas as pd
import pickle
import os
import argparse
from stable_baselines3 import PPO
from custom_env_folder.custom_env2 import Uniswapv3Env

def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained agent on a given data file.")
    parser.add_argument("--data_file", type=str, required=True, help="Path to the data file (CSV) for evaluation.")
    args = parser.parse_args()
    # Hardcoded config
    config = {
        'delta': 0.05,
        'action_values': [0, 30, 60, 90],
        'x': 10,
        'gas_fee': 5
    }
    
    # Window 9 (last model)
    window_num = 9
    expname = "20250726_PPOUniswap_CVRFE"
    
        
    # 2. Load the data
    print("2. Loading test data...")
    uni_test = pd.read_csv(args.data_file)
    print(f"Test data shape: {uni_test.shape}")
    
    # 3. Load the features from the cvrfe and the rf model
    print("3. Loading RF model...")
    vol_model_path = f"src/output/{expname}/rf_models/vol_rfe_model_{window_num}.pkl"
    with open(vol_model_path, 'rb') as f:
        vol_rfe = pickle.load(f)
    print("RF model loaded")
    
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
    
    
    combined_features = ['ma24', 'ma168', 'bb_width_20', 'adxr', 'price_momentum_24', 'aroon_osc_25', 'cp_prob', 'vol_pred', 'ema_5', 'dx']            
    print(f"Updated combined features: {combined_features}")
    print(f"Combined features to use: {combined_features}")
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
    model_path = f"src/output/{expname}/PPOpolicy_{window_num}.zip"
    rlmodel = PPO.load(model_path)
    print("Policy loaded")
    
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
    print(f"Total Steps: {step_count}")
    print(f"Total Reward: {total_reward:.4f}")
    print(f"Average Reward: {total_reward/step_count:.4f}")
    
    # Save results
    test_history = test_env.history
    results_df = pd.DataFrame(test_history)
    results_df.to_csv("evaluation_results.csv", index=False)
    print("Results saved to evaluation_results.csv")

if __name__ == "__main__":
    main()
