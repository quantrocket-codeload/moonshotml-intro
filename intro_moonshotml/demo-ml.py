# Copyright 2020-2024 QuantRocket LLC - All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pandas as pd
from moonshot import MoonshotML
from moonshot.commission import PerShareCommission

class USStockCommission(PerShareCommission):
    BROKER_COMMISSION_PER_SHARE = 0.005

class DemoMLStrategy(MoonshotML):

    CODE = "demo-ml"
    DB = "usstock-free-1d"
    DB_FIELDS = ["Open", "Close"]
    UNIVERSES = "usstock-free"
    LOOKBACK_WINDOWS = [1,2,3,4,5,6,7,8,9,10,12,14,16,18,20,30,40,50,60,80,100,125,150,175,200]
    LOOKBACK_WINDOW = max(LOOKBACK_WINDOWS) # set lookback to longest moving average; more on lookback windows: https://www.quantrocket.com/docs/#moonshot-lookback-windows
    FORWARD_RETURNS_WINDOW = 22
    TOP_N = 3 # number of positions to hold
    COMMISSION_CLASS = USStockCommission
    REBALANCE_INTERVAL = "M" # M = monthly; see https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases

    def prices_to_features(self, prices):
        """
        From a DataFrame of prices, return a tuple of features and targets to be
        provided to the machine learning model.

        The returned features can be a list or dict of DataFrames, where each
        DataFrame is a feature and should have the same shape, with a Date index
        and sids as columns.

        The returned targets should be a DataFrame with an index
        matching the index of the features DataFrames. Targets are
        used in training and are ignored for prediction.
        """
        closes = prices.loc["Close"]
        opens = prices.loc["Open"]

        # FEATURES
        features = {}

        for n in self.LOOKBACK_WINDOWS:
            features[f'return_{n}'] = closes.pct_change(n)

        # TARGET
        returns = opens.pct_change(self.FORWARD_RETURNS_WINDOW)

        # Calculate median cross-sectional returns (a Series)...
        median_returns = returns.median(axis=1)
        # ...and broadcast back to shape of original DataFrame
        median_returns = closes.apply(lambda x: median_returns)

        # Find stocks which will outperfom in the future
        outperformers = returns > median_returns
        targets = outperformers.shift(-self.FORWARD_RETURNS_WINDOW).fillna(False).astype(int)

        # when Moonshot calls predict(), we want it to actually call predict_proba()
        # see https://www.quantrocket.com/docs/#ml-predict-probabilities
        if self.model:
            self.model.predict = self.model.predict_proba

        return features, targets

    def predictions_to_signals(self, predictions, prices):
        """
        From a DataFrame of predictions produced by a machine learning model,
        return a DataFrame of signals. By convention, signals should be
        1=long, 0=cash, -1=short.

        The index of predictions will match the index of the features
        DataFrames returned in prices_to_features.
        """
        # Rank by probability of outperforming
        winner_ranks = predictions.rank(axis=1, ascending=False)

        signals = winner_ranks <= self.TOP_N
        signals = signals.astype(int)

        # Resample using the rebalancing interval.
        # Keep only the last signal of the month, then fill it forward
        signals = signals.resample(self.REBALANCE_INTERVAL).last()
        signals = signals.reindex(predictions.index, method="ffill")

        return signals

    def signals_to_target_weights(self, signals: pd.DataFrame, prices: pd.DataFrame):
        """
        This method receives a DataFrame of integer signals (-1, 0, 1) and
        should return a DataFrame indicating how much capital to allocate to
        the signals, expressed as a percentage of the total capital allocated
        to the strategy (for example, -0.25, 0, 0.1 to indicate 25% short,
        cash, 10% long).
        """
        weights = self.allocate_equal_weights(signals)
        return weights

    def target_weights_to_positions(self, weights: pd.DataFrame, prices: pd.DataFrame):
        """
        This method receives a DataFrame of allocations and should return a
        DataFrame of positions. This allows for modeling the delay between
        when the signal occurs and when the position is entered, and can also
        be used to model non-fills.
        """
        # Enter the position in the period/day after the signal
        return weights.shift()

    def positions_to_gross_returns(self, positions: pd.DataFrame, prices: pd.DataFrame):
        """
        This method receives a DataFrame of positions and a DataFrame of
        prices, and should return a DataFrame of percentage returns before
        commissions and slippage.
        """
        # We'll enter on the open, so our return is today's open to
        # tomorrow's open
        opens = prices.loc["Open"]
        # The return is the security's percent change over the period,
        # multiplied by the position.
        gross_returns = opens.pct_change() * positions.shift()
        return gross_returns
