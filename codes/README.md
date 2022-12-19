# Codes

Here we provide our codes for this project.

## Q-Learner: Main model part:

`QLearner.py`: Implement QLearner, a Reinforcement Learning class;

`StrategyLearner.py`: Implement a StrategyLearner that trains a QLearner for trading a symbol.

- `StrategyLearner.py`: Designed strategy by using effective intraday technical Indicators only;

- `StrategyLearner2.py`: Combined `StrategyLearner.py` with classical momentum indicators;

- `StrategyLearner3.py`: Combined `StrategyLearner2.py` with fundamental indicators;

- `StrategyLearner4.py`: change the way of boolean sequence mapping

`grading.py`: Main grader class; an instance is passed in through a pytest fixture

`grade_strategy_learner.py`: Testing code for StrategyLearner, mostly based on the code provided by Georgia Tech, with the following updates:

- code edits to make the code compatible with Python 3.x
- a lot of reformatting to make code readable and understandable

## Data and Portfolio analysis

`indicators.py`: Calculate the technical indicators we use;

`analysis.py`: Analyze a portfolio; generate risk matrices;

`marketsim.py`: Implement a market simulator that processes a dataframe instead of 
a csv file.

## Notebooks

`q_learning_trading_example.ipynb`: sample code of how strategy learner works

`results.ipynb`: Accumulate all the results by strategies

## Others

`utils`: some util functions for StrategyLearner.

