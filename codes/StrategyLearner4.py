"""Implement a StrategyLearner that trains a QLearner for trading a symbol."""

import numpy as np
import datetime as dt
import pandas as pd
import os
import json
import matplotlib.pyplot as plt

from util import get_data, get_feature_file, create_df_benchmark, bi_to_decimal
import QLearner as ql
from indicators import get_momentum, get_sma_indicator, compute_bollinger_value

import marketsim
import importlib
importlib.reload(marketsim)
importlib.reload(ql)

from marketsim import compute_portvals_single_symbol, market_simulator
from analysis import get_portfolio_stats


class StrategyLearner4(object):
    # Constants for positions and order signals
    LONG = 1
    CASH = 0
    SHORT = -1

    def __init__(self, feature_list, num_shares=1000, epochs=100, min_epoch=50, num_steps=5,
                 impact=0.0, commission=0.00, verbose=False, **kwargs):
        """Instantiate a StrategLearner that can learn a trading policy.

        Parameters:
        num_shares: The number of shares that can be traded in one order
        epochs: The number of times to train the QLearner
        num_steps: The number of steps used in getting thresholds for the
        discretization process. It is the number of groups to put data into.
        impact: The amount the price moves against the trader compared to the
        historical data at each transaction
        commission: The fixed amount in dollars charged for each transaction
        verbose: If True, print and plot data in add_evidence
        **kwargs: Arguments for QLearner
        """
        self.technicals = feature_list
        self.epochs = epochs
        self.min_epoch = min_epoch
        self.num_steps = num_steps
        self.num_shares = num_shares
        self.impact = impact
        self.commission = commission
        self.verbose = verbose
        # Initialize a QLearner
        self.q_learner = ql.QLearner(**kwargs)
        self.MAX = 6
        with open("../Data/symbol_dict.txt") as f:
            self.symbol_dict = json.load(f) 
        
        HealthCare, Financials, InfoTech, ConsumerStaples = self.get_thresholds()

        self.datafull = pd.DataFrame()

        symbols = os.listdir("../Data/financials")
        for _ in symbols:
            symbol = _.split(".")[0]
            # Get adjusted close prices for symbol
            df_features = get_feature_file("financials", symbol)
            
            # if len of df_features < 1 continue
            if len(df_features) < 1:
                continue

            if symbol not in self.symbol_dict:
                continue
            elif self.symbol_dict[symbol] == "HealthCare":
                thresholds = HealthCare
            elif self.symbol_dict[symbol] == "Financials":
                thresholds = Financials
            elif self.symbol_dict[symbol] == "InfoTech":
                thresholds = InfoTech
            elif self.symbol_dict[symbol] == "ConsumerStaples":
                thresholds = ConsumerStaples

            self.discretize(df_features, thresholds)

            # df_features['returns'] = prices.pct_change()
            self.datafull = pd.concat([self.datafull, df_features], axis=0)

        self.datafull.reset_index(inplace=True)

    def get_thresholds(self, start_date=None, end_date=None):
        """Compute the thresholds to be used in the discretization of features.
        thresholds is a 2-d numpy array where the first dimesion indicates the 
        indices of features in df_features and the second dimension refers to 
        the value of a feature at a particular threshold.
        """

        technicals = ['RDSkew','down_vol_ratio','late_trans_ratio','vol_price_corr','avg_trans_outflow_ratio','large_order_drives_gain']
        
        # read HealthCare, Financials, InfoTech, ConsumerStaples
        HealthCare = pd.read_csv("../Data/HealthCare.csv", index_col='Datetime',
            parse_dates=True, na_values=['nan'])
        Financials = pd.read_csv("../Data/Financials.csv", index_col='Datetime',
            parse_dates=True, na_values=['nan'])      
        InfoTech = pd.read_csv("../Data/InfoTech.csv", index_col='Datetime',
            parse_dates=True, na_values=['nan'])
        ConsumerStaples = pd.read_csv("../Data/ConsumerStaples.csv", index_col='Datetime',
            parse_dates=True, na_values=['nan'])

        df = pd.read_csv("../Data/average.csv", index_col='Datetime',
        parse_dates=True, na_values=['nan'])

        if start_date and end_date:
            df = df.loc[start_date:end_date]
            HealthCare = HealthCare.loc[start_date:end_date]
            Financials = Financials.loc[start_date:end_date]
            InfoTech = InfoTech.loc[start_date:end_date]
            ConsumerStaples = ConsumerStaples.loc[start_date:end_date]

        df = df.rolling(window=self.num_steps, min_periods=1).mean()
        HealthCare = HealthCare.rolling(window=self.num_steps, min_periods=1).mean()
        Financials = Financials.rolling(window=self.num_steps, min_periods=1).mean()
        InfoTech = InfoTech.rolling(window=self.num_steps, min_periods=1).mean()
        ConsumerStaples = ConsumerStaples.rolling(window=self.num_steps, min_periods=1).mean()

        HealthCare[technicals] = df[technicals]
        Financials[technicals] = df[technicals]
        InfoTech[technicals] = df[technicals]
        ConsumerStaples[technicals] = df[technicals]

        return HealthCare, Financials, InfoTech, ConsumerStaples

    def discretize(self, df_features, thresholds):
        """Discretize features and return a state.

        Parameters:
        df_features: The technical indicators to be discretized. They were  
        computed in get_features()
        non_neg_position: The position at the beginning of a particular day,
        before taking any action on that day. It is >= 0 so that state >= 0

        Returns:
        state: A state in the Q-table from which we will query for an action.
        It indicates an index of the first dimension in the Q-table
        """
        thresholds[self.technicals] = df_features[self.technicals].rolling(window=self.num_steps, min_periods=1).mean()

        df_features[thresholds.keys()] = df_features[thresholds.keys()] - thresholds
        for key in thresholds.keys():
            df_features.loc[df_features[key]<0, key] = 0
            df_features.loc[df_features[key]>0, key] = 1

        df_features['DR'] = 1 - df_features['DR'] 

        # exchange 0 and 1 for negative effective features
        for key in ['RDSkew','late_trans_ratio','vol_price_corr','large_order_drives_gain']:
            df_features[key] = 1- df_features[key]

        # exchange 0 and 1 for negative effective features
        for key in ['RDSkew','late_trans_ratio','vol_price_corr','large_order_drives_gain']:
            df_features[key] = 1- df_features[key]

    def get_position(self, old_pos, signal):
        """Find a new position based on the old position and the given signal.
        signal = action - 1; action is a result of querying a state, which was
        computed in discretize(), in the Q-table. An action is 0, 1 or 2. It is
        an index of the second dimension in the Q-table. We have to subtract 1
        from action to get a signal of -1, 0 or 1 (short, cash or long).
        """
        new_pos = self.CASH
        # If old_pos is not long and signal is to buy, new_pos will be long
        if signal == 0 : return new_pos
        elif old_pos == 0: new_pos = signal
        elif np.abs(old_pos) < self.MAX and np.sign(signal) == np.sign(old_pos):
            new_pos = signal
        # If old_pos is not short and signal is to sell, new_pos will be short
        elif np.abs(old_pos) < self.MAX and np.sign(signal) != np.sign(old_pos):
            new_pos = -old_pos
        return new_pos

    def get_reward(self, prev_price, curr_price, position):
        """Calculate the daily reward as a percentage change in prices: 
        - Position is long: if the price goes up (curr_price > prev_price),
          we get a positive reward; otherwise, we get a negative reward
        - Position is short: if the price goes down, we get a positive reward;
        otherwise, we a negative reward
        - Position is cash: we get no reward
        """
        return position * ((curr_price / prev_price) - 1)

    def has_converged(self, cum_returns, patience=20):
        """Check if the cumulative returns have converged.

        Paramters:
        cum_returns: A list of cumulative returns for respective epochs
        patience: The number of epochs with no improvement in cum_returns

        Returns: True if converged, False otherwise
        """
        # The number of epochs should be at least patience before checking
        # for convergence
        if patience > len(cum_returns):
            return False
        latest_returns = cum_returns[-patience:]
        # If all the latest returns are the same, return True
        if len(set(latest_returns)) == 1:
            return True
        max_return = max(cum_returns)
        if max_return in latest_returns:
            # If one of recent returns improves, not yet converged
            if max_return not in cum_returns[:len(cum_returns) - patience]:
                return False
            else:
                return True
        # If none of recent returns is greater than max_return, it has converged
        return True

    def add_evidence(self, start_date=dt.datetime(2021,1,1),
        end_date=dt.datetime(2021,12,31), start_val = 10000):
        """Create a QLearner, and train it for trading.

        Parameters:
        symbol: The stock symbol to act on
        start_date: A datetime object that represents the start date
        end_date: A datetime object that represents the end date
        start_val: Start value of the portfolio which contains only the symbol
        """
        
        cum_returns = []
        for epoch in range(1, self.epochs + 1):
            #generate 3000 random numbers between 0 and len(self.datafull)/
            random_nums = np.random.randint(0, len(self.datafull), 3000)

            df_sub = self.datafull.iloc[random_nums].copy()
            df_sub.dropna(inplace=True)
            df_sub = df_sub[(df_sub['Datetime'] >= start_date) & (df_sub['Datetime'] <= end_date)]
            df_sub.reset_index(inplace=True)

            self.q_learner.reset_rar()
            cum_return = 0
            # Initial position is holding nothing
            position = self.CASH
            # Create a series that captures order signals based on actions taken
            orders = pd.Series(np.zeros(len(df_sub)), index=df_sub.index)
            
            # Iterate over the data by date
            for nrow, row in df_sub.iterrows():
                if random_nums[nrow] + 1 >= len(self.datafull): continue

                # Get a state; add 1 to position so that states >= 0
                state = np.array(row.drop(['index', 'Symbol', 'Datetime', 'Close','Returns'])).astype(int)

                action = self.q_learner.query_set_state(bi_to_decimal(state))

                aweek_later = self.datafull.iloc[row['index'] + 1]

                if aweek_later.Symbol != row.Symbol: continue

                reward = action*aweek_later.Returns
                if np.isnan(reward): continue

                cum_return += reward

                try:
                    next_state = np.array(aweek_later.drop(['Symbol', 'Datetime', 'Close','Returns'])).astype(int)
                except:
                    continue

                self.q_learner.query(bi_to_decimal(next_state), reward)
                # # On the first day, get an action without updating the Q-table
                # if timestamp == df_features.index[0]:
                #     action = self.q_learner.query_set_state(state)
                # # On other days, calculate the reward and update the Q-table
                # else:
                #     prev_price = prices.iloc[num-1]
                #     curr_price = prices.loc[timestamp]
                #     reward = self.get_reward(prev_price, curr_price, position)
                    # action = self.q_learner.query(state, reward)
                # On the last day, close any open positions
                # if timestamp == df_features.index[-1]:
                #     new_pos = -position
                # else:
                #     new_pos = self.get_position(position, action - 1)

                # Add new_pos to orders
                orders.iloc[nrow] = action*self.num_shares
                # # Update current position
                # position += new_pos
            
            # df_trades = create_df_trades(orders, self.num_shares)
            # portvals = compute_portvals_single_symbol(df_orders=df_trades, 
            #                                           symbol=symbol, 
            #                                           start_val=start_val, 
            #                                           commission=self.commission,
            #                                           impact=self.impact)
            # cum_return = get_portfolio_stats(portvals)[0]
            
            cum_returns.append(cum_return)
            if self.verbose: 
                print (epoch, cum_return)
            # Check for convergence after running for at least 20 epochs
            if epoch > self.min_epoch:
                # Stop if the cum_return doesn't improve for 10 epochs
                if self.has_converged(cum_returns):
                    break
        if self.verbose:
            plt.plot(cum_returns)
            plt.xlabel("Epoch")
            plt.ylabel("Cumulative return (%)")
            plt.savefig("../Figure/epochs.png")
            plt.show()

    def test_policy(self, symbol, start_date=dt.datetime(2022,1,1),
        end_date=dt.datetime(2022,3,1), start_val=10000):
        """Use the existing policy and test it against new data.

        Parameters:
        symbol: The stock symbol to act on
        start_date: A datetime object that represents the start date
        end_date: A datetime object that represents the end date
        start_val: Start value of the portfolio which contains only the symbol
        
        Returns:
        df_trades: A dataframe whose values represent trades for each day: 
        +1000 indicating a BUY of 1000 shares, and -1000 indicating a SELL of 
        1000 shares
        """
        
        # # Get adjusted close pricess for symbol
        # df_prices = get_feature_file(symbol, start_date, end_date)

        df_test = self.datafull[(self.datafull['Symbol'] == symbol) & (self.datafull['Datetime'] >= start_date) & (self.datafull['Datetime'] <= end_date)]

        # Initial position is holding nothing
        position = self.CASH

        df_test.set_index('Datetime', inplace=True)

        # Create a series that captures order signals based on actions taken
        orders = pd.Series(np.zeros(len(df_test)), index=df_test.index)
        # Iterate over the data by date
        for date in df_test.index:
            
            # Get a state; add 1 to position so that states >= 0
            state = df_test.loc[date].drop([ 'Symbol', 'Close','Returns'])
            if state.isna().any(): continue
            state = np.array(state).astype(int)

            action = self.q_learner.query_set_state(bi_to_decimal(state))
            # On the last day, close any open positions
            if date == df_test.index[-1]:
                new_pos = -position
            else:
                new_pos = self.get_position(position, action)
            # Add new_pos to orders
            orders.loc[date] = new_pos*self.num_shares
            # Update current position
            position += new_pos
        # Create a trade dataframe
        # df_trades = create_df_trades(orders, symbol, self.num_shares)
        
        return orders, df_test['Close']
        

if __name__=="__main__":
    start_val = 100000
    symbol = "JPM"
    commission = 0.00
    impact = 0.0
    num_shares = 1000

    # In-sample or training period
    start_date = dt.datetime(2008, 1, 1)
    end_date = dt.datetime(2009, 12, 31)
    
    # Get a dataframe of benchmark data. Benchmark is a portfolio starting with
    # $100,000, investing in 1000 shares of symbol and holding that position
    df_benchmark_trades = create_df_benchmark(symbol, start_date, end_date, 
                                              num_shares)

    # Train and test a StrategyLearner
    stl = StrategyLearner(num_shares=num_shares, impact=impact, 
                          commission=commission, verbose=True,
                          num_states=3000, num_actions=3)
    stl.add_evidence(symbol=symbol, start_val=start_val, 
                     start_date=start_date, end_date=end_date)
    df_trades = stl.test_policy(symbol=symbol, start_date=start_date,
                                end_date=end_date)
    portvals = compute_portvals_single_symbol(df_orders=df_trades, 
                                              symbol=symbol, 
                                              start_val=start_val, 
                                              commission=0,
                                              impact=0)
    cum_return = get_portfolio_stats(portvals)[0]
    print("Cummulative Return ", cum_return)

    # Retrieve performance stats via a market simulator
    print ("Performances during training period for {}".format(symbol))
    print ("Date Range: {} to {}".format(start_date, end_date))
    market_simulator(df_trades, df_benchmark_trades, symbol=symbol, 
                     start_val=start_val, commission=commission, impact=impact)

    # Out-of-sample or testing period: Perform similiar steps as above,
    # except that we don't train the data (i.e. run add_evidence again)
    start_date = dt.datetime(2010, 1, 1)
    end_date = dt.datetime(2011, 12, 31)
    df_benchmark_trades = create_df_benchmark(symbol, start_date, end_date, 
                                              num_shares)
    df_trades = stl.test_policy(symbol=symbol, start_date=start_date, 
                                end_date=end_date)
    print ("\nPerformances during testing period for {}".format(symbol))
    print ("Date Range: {} to {}".format(start_date, end_date))
    market_simulator(df_trades, df_benchmark_trades, symbol=symbol, 
                     start_val=start_val, commission=commission, impact=impact)