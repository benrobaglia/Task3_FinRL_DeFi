# FinRL DeFi Project - Uniswap Trading Agent

Winner – 1st place, FinAI Contest 2025 (Task 3), IEEE International Conference on Cyber Security and Cloud Computing.

This repository contains the code to reproduce results for the paper:

**DeFi Liquidity Provisioning with Reinforcement Learning and Dynamic Feature Selection**

Submitted for the FinAI Contest 2025 Task 3.

The implementations include Reinforcement Learning agents for trading on Uniswap, with:
- A baseline PPO implementation
- An enhanced version with CVRFE (Cross-Validation Recursive Feature Elimination) and walking forward validation

## Project Structure

Key files:
- `src/uniswap_test.py`: Baseline PPO implementation from the FinAI starter kit (runs with `src/config/uniswap_rl_param_1108.yaml`).
- `src/uniswap_cvrfe_ppo.py`: Main script to reproduce CVRFE-PPO results (runs with `src/config/uniswap_rl_param_2707.yaml`).
- `src/custom_env_folder/custom_env2.py`: Augmented environment with improved state feature handling.
- `src/config/`: Contains configuration YAML files for different experiments.
- `src/output/`: Stores trained policy weights and results.
- `src/data/`: Contains historical test data.

## Setup

1. Clone this repository:
```bash
git clone [repository_url]
cd Task_3_FinRL_DeFi
```

2. Install dependencies:
```bash
pip install -r src/requirements.txt
```

## Running the Baseline PPO

To run the baseline PPO implementation:
```bash
python src/uniswap_test.py
```

This will:
1. Train the PPO agent using the specified configuration
2. Save the trained policy weights to `src/output/[experiment_name]/PPOpolicy.zip`
3. Generate test results in `src/data/[experiment_name]/`

## Running the CVRFE PPO Implementation

To run the enhanced implementation with walking forward validation:
```bash
python src/uniswap_cvrfe_ppo.py
```

This will:
1. Perform walking forward validation with feature selection using CVRFE
2. Train separate policies for each window through two optimization studies:
   - Study 1: Optimizes action values, learning rate, and network architecture
   - Study 2: Optimizes entropy coefficient, gamma, and clip range
3. Save outputs in the following structure:
   - Policy weights: `src/output/[expname]/PPOpolicy_{window}.zip`
   - Test histories: `src/data/[expname]/test_history_{window}.csv`
   - Feature importance: `src/data/[expname]/vol_feature_importance_{window}.csv`
   - Study results: `src/output/[expname]/study_result_rolling_{window}_{study_num}.csv` (0 for Study 1, 1 for Study 2)

## Environment Details

The `custom_env2.py` environment improves upon the original `custom_env.py` by:
- Computing and storing features in a dataframe to prevent misalignment
- Providing an augmented state space for better learning
- Using the configuration from `uniswap_rl_param_2707.yaml`

## Output Structure

Results are organized by experiment date in:
- `src/output/[expname]/`: Contains trained policies and study results
  - `Policy_{window}.zip`: Saved policy weights for each validation window
  - `study_result_rolling_*.csv`: Training metrics
- `src/data/[expname]/`: Contains test histories and feature importance results

## Evaluating a Trained Agent 

The `src/evaluate_agent.py` script is used to evaluate a pre-trained reinforcement learning agent on a specified dataset. This script loads a trained policy and runs it on the provided market data to assess its performance.

### Usage

```bash
python src/evaluate_agent.py --data_file path/to/your/evaluation_data.csv [--window WINDOW_NUM] [--expname EXPERIMENT_NAME]
```

### Arguments

- `--data_file`: (Required) Path to the evaluation data file (CSV format)
- `--window`: (Optional) Window number to use (0-15, default: 15). This selects which trained model to evaluate. **Window 15 is the default as it represents the model trained on the latest available data.**
- `--expname`: (Optional) Experiment name (default: "20250807_PPOUniswap_CVRFE")

### Example

```bash
# Evaluate using the default window 15 model (latest data)
python src/evaluate_agent.py --data_file src/test_set.csv

# Evaluate using a specific window model
python src/evaluate_agent.py --data_file src/test_set.csv --window 9

# Evaluate using a different experiment
python src/evaluate_agent.py --data_file src/test_set.csv --window 5 --expname 20250723_PPOUniswap_RFE
```

### What the script does:

1. **Loads the best configuration** from study results for the specified window
2. **Loads the trained models**: PPO policy, Random Forest feature selection model, and feature information
3. **Processes the evaluation data** through the same feature engineering pipeline used during training
4. **Computes volatility predictions** using the trained Random Forest model
5. **Creates the test environment** with the processed features and configuration
6. **Runs the trained PPO agent** on the evaluation data deterministically
7. **Saves evaluation results** to `evaluation_results_window_{WINDOW_NUM}.csv`

The script automatically handles different model formats and paths, and provides detailed logging of the evaluation process including step-by-step rewards and final performance metrics.

## Configuration Files

Key configuration files:
- `uniswap_rl_param_1108.yaml`: For baseline PPO
- `uniswap_rl_param_2707.yaml`: For CVRFE PPO with walking forward validation
