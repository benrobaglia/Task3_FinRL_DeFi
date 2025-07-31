import numpy as np
import pandas as pd
from custom_env_folder.custom_env import Uniswapv3Env as OriginalEnv
from custom_env_folder.custom_env2 import Uniswapv3Env as AugmentedEnv

def compare_environments():
    # Load sample data
    data = pd.read_csv('data_price_uni_h_time.csv')
    sample_data = data.iloc[:1000]  # Use first 1000 rows for testing
    
    # Initialize both environments with same parameters
    delta = 0.05
    action_values = [0, 20, 50]
    x = 10
    gas = 5
    
    # import market data
    uni_table = pd.read_csv('data_price_uni_h_time.csv')

    uni_time = uni_table[['timestamp']]

    uni_test = uni_table.iloc[1500:3000].reset_index(drop=True)
    uni_test_prices = uni_test[['price']]
    
    orig_env = OriginalEnv(delta=delta, 
                          action_values=action_values,
                          market_data=uni_test_prices,
                          x=x,
                          gas=gas)
                          
    aug_env = AugmentedEnv(delta=delta,
                          action_values=action_values,
                          market_data=uni_test,
                          market_features=None,
                          x=x,
                          gas=gas,
                          process_features=True)
    
    # Reset both environments
    orig_state, _ = orig_env.reset()
    aug_state, _ = aug_env.reset()
    
    # Define the features from original env we want to compare
    orig_features = ['price', 'tick', 'width', 'sigma',
                   'ma24', 'ma168', 'bb_upper', 'bb_middle', 'bb_lower',
                   'ADXR', 'BOP', 'DX']
    
    print("Comparing environments for 10 steps...")
    print("="*50)
    
    for step in range(10):
        # Take random action
        action = np.random.randint(0, len(action_values))
        
        # Step both environments
        orig_state, orig_reward, orig_done, orig_trunc, _ = orig_env.step(action)
        aug_state, aug_reward, aug_done, aug_trunc, _ = aug_env.step(action)
        
        # Extract comparable features from augmented state
        comparable_aug_state = np.delete(aug_state, 5) # Remove liquidity ratio
                
        # Compare states
        state_diff = np.array(orig_state) - np.array(comparable_aug_state)
        
        print(f"\nStep {step + 1}:")
        print(f"Action taken: {action_values[action]}")
        print("Original vs Augmented Rewards:", orig_reward, aug_reward)
        print("State differences (original - augmented):")
        for feat, diff in zip(orig_features, state_diff):
            print(f"{feat}: {diff:.6f}")
        
        if orig_done or aug_done:
            print("Environment terminated early")
            break
            
    print("\nComparison complete")

if __name__ == "__main__":
    compare_environments()
