# FinRL DeFi Project - Uniswap Trading Agent

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
python src/uniswap_cvrfe_ppo.py --config src/config/uniswap_rl_param_2707.yaml
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

To evaluate an agent, run the script with the `--data_file` argument pointing to your evaluation data (CSV format):

```bash
python src/evaluate_agent.py --data_file path/to/your/evaluation_data.csv
```

This will:
1. Load the specified data file.
2. Load the pre-trained policy (configured within the script).
3. Run the policy on the evaluation data.
4. Save the evaluation results to `evaluation_results.csv`.

## Configuration Files

Key configuration files:
- `uniswap_rl_param_1108.yaml`: For baseline PPO
- `uniswap_rl_param_2707.yaml`: For CVRFE PPO with walking forward validation
