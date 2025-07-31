'''
MODIFIED VERSION OF uniswap_cvrfe_ppo.py TO ONLY RUN RFE AND SAVE MODELS

This script extracts only the Random Forest feature selection part from the main training script.
Use this when you only want to generate and save the RF models without running the full RL training.

WHAT THIS SCRIPT DOES:
1. Loads and processes the data
2. Performs CV-RFE feature selection using Random Forest
3. Saves the trained RF models and feature information
4. Skips all the RL training (Optuna optimization, PPO training, etc.)

USAGE:
python rfe_only_script.py

OUTPUTS:
- RF models saved to: output/{expname}/rf_models/vol_rfe_model_{i}.pkl
- Feature info saved to: output/{expname}/rf_models/features_info_{i}.pkl
- Feature importance CSV: data/{expname}/vol_feature_importance_{i}.csv
'''

import numpy as np
import pandas as pd
import os
import pickle
import yaml
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFECV
from custom_env_folder.custom_env2 import Uniswapv3Env

def perform_cvrfe_rf(train_df, features, target_column, min_features_to_select=5):
    """
    Perform CV-RFE for automatic feature selection using Random Forest Regressor.
    Includes proper data cleaning to handle infinity and NaN values.
    """
    # Drop rows with NaN in features or target
    df_clean = train_df.dropna(subset=features + [target_column])
    
    # Handle infinity values - replace with NaN and then drop
    df_clean = df_clean.replace([np.inf, -np.inf], np.nan)
    df_clean = df_clean.dropna(subset=features + [target_column])
    
    # Additional check for remaining problematic values
    X = df_clean[features]
    y = df_clean[target_column]
    
    # Check for any remaining infinite or very large values
    if np.any(np.isinf(X.values)) or np.any(np.isnan(X.values)):
        print("Warning: Found infinite or NaN values in features after cleaning")
        # Fill any remaining NaN with median values
        X = X.fillna(X.median())
        # Clip extreme values
        X = X.clip(-1e10, 1e10)
    
    if np.any(np.isinf(y.values)) or np.any(np.isnan(y.values)):
        print("Warning: Found infinite or NaN values in target after cleaning")
        y = y.fillna(y.median())
        y = y.clip(-1e10, 1e10)
    
    print(f"Data shape after cleaning: X={X.shape}, y={y.shape}")
    
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)

    # CV-RFE setup: uses cross-validation to select optimal number of features
    cvrfe = RFECV(estimator=rf, step=1, cv=5, scoring='neg_mean_squared_error', 
                  min_features_to_select=min_features_to_select, n_jobs=-1)
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

if __name__ == "__main__":
    
    # Create directories
    base_dir = os.getcwd()
    output_dir = os.path.join(base_dir, "output")
    config_dir = os.path.join(base_dir, "config")
    data_dir = os.path.join(base_dir, "data")
    
    dirs = [output_dir, config_dir, data_dir]
    for dir in dirs:
        if not os.path.exists(dir):
            os.makedirs(dir)  
    
    # Load hyperparam file
    with open(os.path.join(config_dir, "uniswap_rl_param_2707.yaml"), "r") as f:
        params = yaml.safe_load(f)
    
    # Import market data
    uni_table = pd.read_csv(params['filename'])
    
    # Split data into rolling windows (same as original script)
    split_size = 1500
    dfs_list = [uni_table.iloc[i:i + split_size].reset_index(drop=True) for i in range(0, len(uni_table), split_size)]
    dfs_list.pop()  # drop the last df which has different length
    
    # Set experiment name
    expname = "20250726_PPOUniswap_CVRFE" 
    
    print(f"Processing {len(dfs_list)-5} rolling windows for RFE...")
    
    # Process each rolling window
    for i in range(len(dfs_list)-5):
        print(f"\n=== ROLLING WINDOW: {i+1}/{len(dfs_list)-5} ===")
        
        # Prepare training data (5 consecutive windows)
        df0 = dfs_list[i]
        df1 = dfs_list[i+1]
        df2 = dfs_list[i+2]
        df3 = dfs_list[i+3]
        df4 = dfs_list[i+4]
        
        uni_train = pd.concat([df0, df1, df2, df3, df4], ignore_index=True)
        
        # ===== RFE FEATURE SELECTION SECTION =====
        # This is the ONLY part we keep from the original script
        
        # Process features using the environment
        tenv = Uniswapv3Env(
            delta=params['delta'], 
            action_values=params['action_values'], 
            market_data=uni_train, 
            market_features=None,  # Set to None to use default features
            x=params['x'], 
            gas=params['gas_fee'],
            process_features=True  # Important: process features
        )
        
        uni_train_process = tenv.market_data.copy()
        
        # Define features for RFE (exclude non-feature columns)
        features = list(uni_train_process.columns)
        to_remove = ['timestamp', 'price', 'open_price', 'high_price', 'low_price', 'closed_price', 'log_returns']
        for col in to_remove:
            if col in features:
                features.remove(col)
        
        # Create target variable for volatility prediction
        uni_train_process['future_vol'] = uni_train_process['ew_sigma'].shift(-1)
        uni_train_process = uni_train_process.dropna()  # Clean NaNs
        
        print("Performing CV-RFE feature selection...")
        features_vol, vol_feature_importance, vol_rfe = perform_cvrfe_rf(
            uni_train_process, features, 'future_vol', min_features_to_select=3
        )
        
        # Save feature importance CSV
        os.makedirs(f"data/{expname}", exist_ok=True)
        vol_feature_importance.to_csv(f"data/{expname}/vol_feature_importance_{i}.csv", index=False)
        print(f"Saved feature importance to: data/{expname}/vol_feature_importance_{i}.csv")
        
        # Save the Random Forest model
        os.makedirs(f"output/{expname}/rf_models", exist_ok=True)
        vol_rfe_path = f"output/{expname}/rf_models/vol_rfe_model_{i}.pkl"
        with open(vol_rfe_path, 'wb') as f:
            pickle.dump(vol_rfe, f)
        print(f"Saved volatility RF model to: {vol_rfe_path}")
        
        # Prepare combined features (same logic as original)
        market_features_base = ['ma24', 'ma168', 'bb_width_20', 'adxr', 'price_momentum_24', 'aroon_osc_25', 'cp_prob']
        combined_features = list(set(market_features_base) | set(features_vol))
        
        # Remove ew_sigma and add vol_pred
        if 'ew_sigma' in combined_features:
            combined_features.remove('ew_sigma')
        combined_features.append('vol_pred')
        
        # Save complete features information
        features_info = {
            'features_vol': features_vol,
            'combined_features': combined_features,
            'market_features_base': market_features_base,
            'all_features': features
        }
        
        features_info_path = f"output/{expname}/rf_models/features_info_{i}.pkl"
        with open(features_info_path, 'wb') as f:
            pickle.dump(features_info, f)
        print(f"Saved features info to: {features_info_path}")
        
        print(f"Window {i} completed. Selected {len(features_vol)} features: {features_vol}")
    
    print(f"\n=== RFE PROCESSING COMPLETE ===")
    print(f"All RF models and feature information saved to: output/{expname}/rf_models/")
    print(f"Feature importance files saved to: data/{expname}/")
    print("\nYou can now use these saved models with the evaluate_agent.py script!")


# ===== INSTRUCTIONS FOR MODIFYING THE ORIGINAL SCRIPT =====
"""
TO MODIFY THE ORIGINAL uniswap_cvrfe_ppo.py TO ONLY RUN RFE:

1. KEEP THESE SECTIONS:
   - All imports (especially pickle, sklearn)
   - perform_cvrfe_rf() function
   - Directory creation code
   - Data loading and splitting code
   - The RFE section (lines ~380-430 in original)

2. COMMENT OUT OR REMOVE THESE SECTIONS:
   - optimize_ppo() function
   - optuna_study() function  
   - RewardLoggerCallback class
   - evaluate_model() function
   - All the RL training parts:
     * study 1: optimize action_value, learning_rate, etc.
     * study 2: optimize ent_coef, gamma, clip_range
     * study 3: visualization and testing
   
3. MODIFY THE MAIN LOOP:
   - Keep only the RFE section (study 0)
   - Remove the optuna_study() calls
   - Remove the PPO training and testing parts
   - Add print statements for progress tracking

4. EXAMPLE OF WHAT TO COMMENT OUT:
   ```python
   # COMMENT OUT THESE SECTIONS:
   
   # '''
   # study 1: optimize action_value, learning_rate, dim_hidden_layers, activation
   # '''
   # best_performance = -float('inf')
   # best_trial_num = -1
   # trial_counter = 0
   # out_filename = "output/{}/study_result_rolling_{}_0.csv".format(expname,i)
   # trial1 = optuna_study(...)
   
   # '''
   # study 2: optimize ent_coef, gamma, clip_range  
   # '''
   # ... (all the RL training code)
   
   # '''
   # study 3: after optimization, visualize the test data
   # '''
   # ... (all the testing and visualization code)
   ```

5. THE RESULT:
   - Much faster execution (no RL training)
   - Only generates the RF models needed for evaluation
   - Same output format as the full script
   - Can be used with evaluate_agent.py
"""
