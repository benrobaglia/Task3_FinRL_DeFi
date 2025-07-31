# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 13:24:34 2024
@author: ab978
"""
import pdb
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from typing import List



# from hurst import compute_Hc

import talib
from talib import MA_Type

'''
PROBLEM STATEMENT:  a liquidity provider (LP) holds capital and want to participate in
an AMM Uniswap v3 pool. She can adjust the position at discrete hourly time steps.

It is an optimal uniform allocation strategy around the current price of the token pair.

There has to be a mapping: action is discrete integer i, then current price is p. 
The interval will be [current_tick - i, current_tick + i] which then translates to
[p_l,p_u]

state variables:
    - tech idx
    - USD in portfolio
    - width of liquidity interval (previous action)
    - value of liquidity position at t in USD
    - central tick of the price interval
    
Value of liquidity position can be initialized in several ways (varying initial funds)

To map ticks to prices one needs to implement the formula at page 161 of Ottina book

Action space is discrete from 0 to N where N is the max width of the liquidity range allowed.

Reward is the result of liquidity reallocation

'''

def choppiness_index(high, low, close, period=14):
    tr = talib.TRANGE(high, low, close)
    atr_sum = tr.rolling(window=period).sum()
    high_low_range = high.rolling(window=period).max() - low.rolling(window=period).min()
    return 100 * np.log10(atr_sum / high_low_range) / np.log10(period)


class Uniswapv3Env(gym.Env):
    """
    A custom environment for simulating interaction in a Uniswapv3 AMM.
    
    Attributes:
        delta (float): The fee tier of the AMM.
        n_actions (int): Choices for price range width
        market_data (np.ndarray): The preorganized data from a pandas DataFrame, used for simulation.
        d (float): The tick spacing of the AMM.
        x (int): Initial quantity of asset X (ETH)
        gas (float): fixed gas fee
    """
    
    def __init__(self, 
                 delta: float, 
                 action_values: np.array,
                 market_data: pd.DataFrame, # columns: timestamp, price
                 market_features: list,
                 x: int,
                 gas: float,
                 process_features: bool = True
                 ):
        
        super(Uniswapv3Env, self).__init__()
        # store array of preorganized data from a pandas dataframe
        self.delta = delta
        self.features_position = ['price', 'tick', 'width', 'liquidity', 'sigma', 'liquidity_ratio']
        
        if market_features is None:
            market_features = ['ma24', 'ma168', 'bb_upper', 'bb_middle', 'bb_lower', 'adxr', 'bop', 'dx']
        self.market_features = market_features
        
        self.features_list = self.features_position + self.market_features
        if process_features:
            self.market_data = self._create_state_df(market_data)
        else:
            self.market_data = market_data.copy()

        self.d = self._fee_to_tickspacing(self.delta) # tick spacing
        
        # action space
        self.action_space = spaces.Discrete(len(action_values))
        self.action_values = action_values
        
        self.w = self.action_values[1] # initial interval width
        
        # gas fee
        self.gas = gas
        self.x = x
        self.initial_x = x
        
        # Boundaries to choose
        lower_bounds = []
        upper_bounds = []
        for f in self.features_list:
            lower_bounds.append(-np.inf)
            upper_bounds.append(np.inf)
        lower_bounds = np.array(lower_bounds, dtype=np.float32)
        upper_bounds = np.array(upper_bounds, dtype=np.float32)
        # Define the observation space as a continuous vector space
        self.observation_space = spaces.Box(low=lower_bounds, high=upper_bounds, shape=(len(self.features_list),), dtype=np.float32)
        # To test it run "self.observation_space.contains(np.array([1,2,1,1,1,1],dtype=np.float32))"
        
        # Initialize current state
        self.current_state = None
        
            
    def reset(self, **kwargs):
    
        self.history = []
        
        self.count = 0 # iteration counter
        self.cumul_reward = 0
        self.cumul_fee = 0
        
        state = self.market_data.iloc[self.count]
        
        pt = state['price']
        m = self._price2tick(pt)  # Convert price to tick
        
        # initialize liquidity
        tl, tu = m - self.d*self.w, m + self.d*self.w
        pl, pu = self._tick2price(tl), self._tick2price(tu)
        self.pl = pl
        self.pu = pu
        self.x = self.initial_x
        self.l = self.x / (1/np.sqrt(pt) - 1/np.sqrt(pu))
        
        self.y = self.l * (np.sqrt(pt) - np.sqrt(pl))
        
        ma24 = state['ma24']
        ma168 = state['ma168']
        
        # record data
        self.history.append({
            'X': self.x,
            'Y': self.y,
            'Liquidity': self.l,
            'Price': pt,
            'Price_Upper': pu,
            'Price_Lower': pl,
            'Gas': self.gas,
            'Action': self.w,
            'Sigma': 1,
            'Width': self.w,
            'Reward': 0,
            'Fee': 0,
            'LVR': 0,
            'ma24': ma24,
            'ma168': ma168,
            'Value': self.x * pt + self.y,
        })
        
        liquidity_ratio = self.y / self.x if self.x != 0 else 0
        states = [pt, m, self.w, self.l, 1, liquidity_ratio]
        states.extend([state[feature] for feature in self.market_features])
        self.current_state = np.array(states)
        
        return self.current_state, {} # a dict of info is needed and I initialized it empty
    
    def step(self, action_index):
        """
        Advances the environment by one step based on the given action.

        Parameters:
            action (int): The action taken by the external RL agent, representing an interest rate adjustment.

        Returns:
            Tuple[np.ndarray, float, bool, dict]: A tuple containing the next observation, the calculated reward, done flag, and additional info.
        """
        action = self.action_values[action_index]
        
        # 0
        # update count
        self.count += 1
        
        # record xt_1 and yt_1
        xt_1 = self.x
        yt_1 = self.y
        
        # 1
        # calculate tick corresponding to AMM market price
        # pt_1: price 1h before
        # pt:   price now
        state = self.market_data.iloc[self.count]
        state_1 = self.market_data.iloc[self.count-1]

        pt_1, pt = state_1['price'], state['price']
        
        # 2
        # calculate xt and yt
        # based on pt and old price interval
        # without changing liquidity
        m = self._price2tick(pt)
        pl_1, pu_1 = self.pl, self.pu
        xt, yt = self._calculate_xy(pt, pl_1, pu_1)
        
        # 2.1 
        # if xt or yt is already 0 after 1 hour of market evolution
        # we have to reposition the LP
        if xt == 0 or yt == 0:
            # print("pt out of the (pl_1, pu_1)")
            # if action is 0, force action to be 1
            # reset the LP set interval width +/- 1
            if action == 0:
                pl = self.pl
                pu = self.pu
                # calculate liquidity
                if xt == 0 and yt != 0:
                    self.l = yt / (np.sqrt(pu) - np.sqrt(pl))
                elif yt == 0 and xt != 0:
                    self.l = xt / (1/np.sqrt(pl) - 1/np.sqrt(pu))
                else:   # xt = 0 and yt = 0
                    self.l = 0
                
            else:
                # calculate new interval based on action
                self.w = action
                tl, tu = m - self.d*self.w, m + self.d*self.w
                pl, pu = self._tick2price(tl), self._tick2price(tu)  
                
                # calculate new X and Y based pt
                if yt == 0:
                    # print("X: ", xt, " Y:  0", "Reset Y")
                    xt = xt/2
                    yt = xt * pt
                elif xt == 0:
                    # print("X:  0  Y:  ", yt, "Reset X")
                    yt = yt/2
                    xt = yt/pt
                    
                # update liquidity
                self.l = xt / (1/np.sqrt(pt) - 1/np.sqrt(pu))
            
        # 2.2 
        # price not out of old interval
        else:
            if action != 0:
                self.w = action
                # calculate new pl, pu
                tl, tu = m - self.d*self.w, m + self.d*self.w
                pl, pu = self._tick2price(tl), self._tick2price(tu)   

                # (pull all xt, yt out of the pool)
                # inject the same amount of xt and yt back to the pool
                # calculate new liquidity
                self.l = xt / (1/np.sqrt(pt) - 1/np.sqrt(pu))
                
            else:
                # action = 0, we do not need to do anything
                # update pl, pu
                pl = pl_1
                pu = pu_1

        # update self.x and self.y to xt and yt
        self.x = xt
        self.y = yt
        
        # reward as per the original paper
        gas_fee = self._indicator(action)*self.gas
        
        # calculate fees
        if pt_1 <= pt:
            if pt_1 >= pu or pt <= pl:
                fees = 0
            else:
                p_prime = np.minimum(pt, pu)
                p = np.maximum(pt_1, pl)
                fees = self._calculate_fee(p, p_prime)
        else:
            if pt_1 <= pl or pt >= pu:
                fees = 0
            else:
                p_prime = np.maximum(pt, pl)
                p = np.minimum(pt, pu)
                fees = self._calculate_fee(p, p_prime)
        
        sigma = state['ew_sigma']   
        
        vp = self.x * pt + self.y
        ll = self.l * sigma * sigma / 4 * np.sqrt(pt)
        if vp != 0:
            lvr = ll
        else:
            lvr = 1e+9
            
        # if self.x == 0 or self.y == 0:
        #     lvr = 0
        
        # print("fee: ", fees, ", LVR: ", lvr)
        # print("Gas Fee: ", gas_fee, " Fee: ", fees, " LVR: ", lvr)
        reward = - gas_fee + fees - lvr
        self.cumul_reward += reward
        self.cumul_fee += fees
        
        # consider ma24 and ma24*7=168, ma 24*30 and ma 24*60
        # pre-train the moving average and cut the data
        # plot and share the plot on teams
        # retry the hyperparameters without seed varying
        
        # try exponentially weighted moving average
        ma24 = state['ma24']
        ma168 = state['ma168']
        
        self.initial_value = self.x * pt + self.y

        # record data
        self.history.append({
            'X': self.x,
            'Y': self.y,
            'Liquidity': self.l,
            'Price': pt,
            'Price_Upper': pu,
            'Price_Lower': pl,
            'Gas': gas_fee,
            'Action': action,
            'Sigma': sigma,
            'Width': self.w,
            'Reward': reward,
            'Fee': fees,
            'LVR': lvr,
            'ma24': ma24,
            'ma168': ma168,
            'Value': self.initial_value
        })
        
        self.pl = pl
        self.pu = pu

        liquidity_ratio = self.y / self.x if self.x != 0 else 0
        states = [pt, m, self.w, self.l, sigma, liquidity_ratio]
        states.extend([state[feature] for feature in self.market_features])
        self.current_state = np.array(states)
        
        terminated = self.count >= self.market_data.shape[0] - 2
        truncated = self.l <= 1e-9

        info = {}


        return self.current_state, reward, terminated, truncated, info
    
    def _create_state_df(self, market_data):
        data = market_data.copy()
        data['timestamp'] = pd.to_datetime(data['timestamp'])

        # Rolling OHLC (kept as-is for core price data)
        data['open_price'] = data['price'].rolling(12, min_periods=1).apply(lambda x: x.iloc[0])
        data['high_price'] = data['price'].rolling(12, min_periods=1).max()
        data['low_price'] = data['price'].rolling(12, min_periods=1).min()
        data['closed_price'] = data['price'].rolling(12, min_periods=1).apply(lambda x: x.iloc[-1])

        # Returns and volatility
        data['log_returns'] = np.log(data['price'] / data['price'].shift(1))
        data['ew_sigma'] = data['log_returns'].ewm(alpha=0.05).std()

        ma_window_0 = 24
        ma_window_max = 168

        data['ma24'] = data['price'].rolling(ma_window_0).mean()
        data['ma168'] = data['price'].rolling(ma_window_max).mean()

        periods_sma = [20, 50, 100] 
        for period in periods_sma:
            data[f'sma_{period}'] = talib.SMA(data['closed_price'], timeperiod=period)

        periods_ema = [5, 12, 26, 50, 100]
        for period in periods_ema:
            data[f'ema_{period}'] = talib.EMA(data['closed_price'], timeperiod=period)

        bb_upper, bb_middle, bb_lower = talib.BBANDS(data['price'].values, matype=MA_Type.T3)
        # bb_upper, bb_middle, bb_lower = talib.BBANDS(data['price'].values, timeperiod=20, matype=MA_Type.T3)
        data['bb_upper'] = bb_upper
        data['bb_middle'] = bb_middle
        data['bb_lower'] = bb_lower
        data['bb_width_20'] = (bb_upper - bb_lower) / bb_middle
        data['bb_percent_20'] = (data['closed_price'] - bb_lower) / (bb_upper - bb_lower)

        data['adxr'] = talib.ADX(data['high_price'].values, data['low_price'].values, data['closed_price'].values, timeperiod=14)
        data['dx'] = talib.DX(data['high_price'].values, data['low_price'].values, data['closed_price'].values, timeperiod=14)

        periods_rsi = [7, 14, 21] 
        for period in periods_rsi:
            data[f'rsi_{period}'] = talib.RSI(data['closed_price'], timeperiod=period)

        periods_stoch = [5, 14]
        for period in periods_stoch:
            slowk, slowd = talib.STOCH(data['high_price'], data['low_price'], data['closed_price'], fastk_period=period, slowk_period=3, slowd_period=3)
            data[f'stoch_k_{period}'] = slowk
            data[f'stoch_d_{period}'] = slowd

        data['uo'] = talib.ULTOSC(data['high_price'], data['low_price'], data['closed_price'], timeperiod1=7, timeperiod2=14, timeperiod3=28)
        data['cmo_20'] = talib.CMO(data['closed_price'], timeperiod=20)

        periods_atr = [7, 14, 21]
        for period in periods_atr:
            data[f'atr_{period}'] = talib.ATR(data['high_price'], data['low_price'], data['closed_price'], timeperiod=period)

        periods_std = [20]  # Reduced
        for period in periods_std:
            data[f'stddev_{period}'] = talib.STDDEV(data['closed_price'], timeperiod=period)

        data['vix_proxy'] = data['log_returns'].rolling(20).std()

        macd, macdsignal, macdhist = talib.MACD(data['closed_price'], fastperiod=12, slowperiod=26, signalperiod=9)
        data['macd'] = macd
        data['macd_signal'] = macdsignal
        data['macd_hist'] = macdhist 

        data['plus_di'] = talib.PLUS_DI(data['high_price'], data['low_price'], data['closed_price'], timeperiod=14)
        data['minus_di'] = talib.MINUS_DI(data['high_price'], data['low_price'], data['closed_price'], timeperiod=14)

        data['aroon_osc_25'] = talib.AROONOSC(data['high_price'], data['low_price'], timeperiod=25)

        data['sar'] = talib.SAR(data['high_price'], data['low_price'], acceleration=0.02, maximum=0.2)
        data['trix'] = talib.TRIX(data['closed_price'], timeperiod=15)

        data['price_sma20_ratio'] = data['price'] / data['sma_20'] 
        data['price_ema12_ratio'] = data['price'] / data['ema_12'] 

        horizons = [5, 10, 24]
        for horizon in horizons:
            data[f'price_momentum_{horizon}'] = data['price'] / data['price'].shift(horizon) - 1

        data['ema_crossover'] = np.where(data['ema_5'] > data['ema_26'], 1, 0)

        data['chop_14'] = choppiness_index(data['high_price'], data['low_price'], data['closed_price'], period=14)
        data['chop_14'].fillna(method='ffill', inplace=True) 

        data['bop'] = talib.BOP(data['open_price'], data['high_price'], data['low_price'], data['closed_price'])   # Balance Of Power

        data = self._add_bocpd_feature(data)

        return data[ma_window_max:].reset_index(drop=True).ffill()


    
    # def _calculate_candles(self, interval_length: int):
    #     open_prices = []
    #     high_prices = []
    #     low_prices = []
    #     close_prices = []

    #     interval_length = interval_length - 1

    #     # Iterate through market data
    #     for i in range(len(self.market_data)):
    #         if i < interval_length:
    #             # For the first 0-10 (11) data points, calculate the i-hour candle
    #             start_idx = 0
    #         else:
    #             # For the rest, use the last 12 hours of data
    #             start_idx = i - interval_length

    #         # Get the relevant data slice
    #         data_slice = self.market_data[start_idx : i+1]
            
    #         # Calculate open, high, low, and close
    #         open_price = data_slice[0]
    #         high_price = max(data_slice)
    #         low_price = min(data_slice)
    #         close_price = data_slice[-1]

    #         # Append the results to their respective arrays
    #         open_prices.append(open_price)
    #         high_prices.append(high_price)
    #         low_prices.append(low_price)
    #         close_prices.append(close_price)
            
    #     # Convert lists to NumPy arrays with dtype float64
    #     open_prices = np.array(open_prices, dtype=np.float64)
    #     high_prices = np.array(high_prices, dtype=np.float64)
    #     low_prices = np.array(low_prices, dtype=np.float64)
    #     close_prices = np.array(close_prices, dtype=np.float64)

    #     return open_prices, high_prices, low_prices, close_prices

    def _price2tick(self, p: float):
        return math.floor(math.log(p, 1.0001))
    
    def _tick2price(self, t: int):
        return 1.0001**t
    
    def _fee_to_tickspacing(self, fee_tier: float):
        if fee_tier == 0.05:
            return 10
        elif fee_tier == 0.30:
            return 60
        elif fee_tier == 1.00:
            return 200
        else:
            raise ValueError(f"Unsupported fee tier: {fee_tier}")
            
    def _calculate_fee(self, p, p_prime):
        if p <= p_prime:
            # fee = (self.delta / (1 - self.delta)) * self.l * (math.sqrt(p) - math.sqrt(p_prime))
            fee = (self.delta / (1 - self.delta)) * self.l * (math.sqrt(p_prime) - math.sqrt(p))
        else:
            # fee = (self.delta / (1 - self.delta)) * self.l * ((1 / math.sqrt(p)) - (1 / math.sqrt(p_prime))) * p_prime
            fee = (self.delta / (1 - self.delta)) * self.l * ((1 / math.sqrt(p_prime)) - (1 / math.sqrt(p))) * p_prime
        return fee
    
    def _indicator(self,a):
        return 1 if a != 0 else 0
    
    def _calculate_xy(self, p, pl, pu):
        # Ottina et al. Page 169
        if p <= pl:
            x = self.l * (1 / math.sqrt(pl) - 1 / math.sqrt(pu))
            y = 0
        elif p >= pu:
            x = 0
            y = self.l * (math.sqrt(pu) - math.sqrt(pl))
        else:  # pl < p < pu
            x = self.l * (1 / math.sqrt(p) - 1 / math.sqrt(pu))
            y = self.l * (math.sqrt(p) - math.sqrt(pl))
        return x, y
    
    def _calculate_ma(self, array, count, window_size):
        # https://www.geeksforgeeks.org/how-to-calculate-moving-averages-in-python/
        if window_size > count:
            window_size = count
            
        window = array[count - window_size : count]
        window_average = np.round(np.sum(window) / window_size, 2)
        
        return window_average
    
    def _add_bocpd_feature(self, data):
        """
        Adds a simplified, BOCPD-like change point feature using the CUSUM algorithm.
        This method is designed for online detection of shifts in the mean of log returns.
        """
        returns = data['log_returns'].fillna(0).values
        n = len(returns)
        cp_scores = np.zeros(n)

        # CUSUM parameters - these can be tuned
        threshold = returns[returns != 0].std() * 4  # A threshold to signal a change, e.g., 4 standard deviations
        drift = returns[returns != 0].std() * 0.5   # A drift term to make the CUSUM less sensitive to noise
        target_mean = 0.0  # We expect log returns to have a mean of zero in a stable regime

        cusum_pos = 0.0  # CUSUM for detecting positive shifts
        cusum_neg = 0.0  # CUSUM for detecting negative shifts

        for t in range(n):
            # Calculate deviation from the target mean
            deviation = returns[t] - target_mean

            # Update positive CUSUM: accumulates positive deviations, resets if it goes below zero
            cusum_pos = max(0, cusum_pos + deviation - drift)

            # Update negative CUSUM: accumulates negative deviations, resets if it goes above zero
            cusum_neg = min(0, cusum_neg + deviation + drift)

            # A change is detected if either sum exceeds the threshold
            if cusum_pos > threshold or cusum_neg < -threshold:
                cp_scores[t] = 1.0  # A strong signal of a change point
                cusum_pos = 0.0     # Reset after detection to be ready for the next change
                cusum_neg = 0.0

        # Smooth the binary signal to create a "probability-like" feature
        # This helps the model see the "echo" of a change point for a short while after it occurs.
        data['cp_prob'] = pd.Series(cp_scores).ewm(span=20, adjust=False).mean()
        data['cp_prob'].fillna(0, inplace=True)

        return data






class CustomMLPFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self,
                 observation_space: spaces.Box,
                 features_dim: int = 256,
                 hidden_dims: List[int] = [128, 64],
                 activation: str = 'relu',
                 dropout_rate: float = 0.0):
        super().__init__(observation_space, features_dim)

        # Activation function mapping for flexibility
        activation_map = {'relu': nn.ReLU, 'tanh': nn.Tanh, 'leaky_relu': nn.LeakyReLU}
        try:
            act_fn = activation_map[activation.lower()]
        except KeyError:
            raise ValueError(f"Unsupported activation function: {activation}")

        layers = [nn.BatchNorm1d(observation_space.shape[0], affine=True)]
        input_dim = observation_space.shape[0]

        # Dynamically create the hidden layers based on the hidden_dims list
        for layer_size in hidden_dims:
            layers.append(nn.Linear(input_dim, layer_size))
            layers.append(act_fn())
            if dropout_rate > 0.0:
                layers.append(nn.Dropout(p=dropout_rate))
            input_dim = layer_size  # The output of this layer is the input to the next

        # Final layer to output the desired features_dim
        layers.append(nn.Linear(input_dim, features_dim))

        self.net = nn.Sequential(*layers)
        
        # Apply custom weight initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        # Apply a good default weight initialization for linear layers
        if isinstance(module, nn.Linear):
            # Kaiming (He) initialization is often recommended for ReLU activations
            nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.net(observations)
