import yfinance as yf
import datetime as dt
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from log.logger import notify


class StockAnalisys:

    def __init__(self, list_ticker, data_period, training_period):
        self.list_ticker = list_ticker
        self.data_period = data_period
        self.training_period = training_period
        self.fallas = []
        self.df = pd.DataFrame()
        self.returns = None
        self.in_sample = None
        self.tickers = None
        self.covariance_matrix_in_sample = None
        self.eigenportfolio = None
        self.eigenportfolio2 = None
        self.in_sample_ind = None
        self.out_sample_ind = None
        self.cumulative_returns = None
        self.min_var_portfolio = None
        self.eigenportfolio_largest = None
        self.cumulative_returns_largest = None

    def get_data(self):
        """
        self.data_period is the period I want to obtain the data, in this case "W" is weekly
        :return: The data from the YahooFinance API and returns in a DataFrame.
        """
        for stock in self.list_ticker:
            try:
                data = yf.download(stock, start="2015-01-01", end=dt.date.today().strftime(
                    "%Y-%m-%d")).resample(self.data_period).last()
                data.rename(columns={"Adj Close": stock}, inplace=True)
                data.drop(["Open", "High", "Low", "Close", "Volume"], axis=1, inplace=True)
                self.df = self.df.merge(data, right_index=True, left_index=True, how="outer")
                self.returns = self.df.pct_change()
                # self.returns = self.returns.iloc[1:, :]  # Remove first row of NA's
                self.returns = self.returns.dropna()
                self.in_sample = self.returns.iloc[:(self.returns.shape[0] - self.training_period), :]
                self.tickers = self.returns.columns.copy()  # Saving the tickets
                # notify(self.df)
                # notify(self.returns)
                # notify(self.in_sample)
                # notify(self.tickers)
            except Exception as e:
                notify(e)
                self.fallas.append(stock)
                notify(self.fallas)

    def covariance_matrix(self):
        # The Eigenvalues of the Covariance Matrix
        self.covariance_matrix_in_sample = self.in_sample.cov().values
        inv_cov_mat = np.linalg.pinv(self.covariance_matrix_in_sample)

        # D, S = np.linalg.eigh(self.covariance_matrix_in_sample)
        # eigenportfolio_1 = S[:, -1] / np.sum(S[:, -1])  # Normalize to sum to 1
        # eigenportfolio_2 = S[:, -2] / np.sum(S[:, -2])  # Normalize to sum to 1
        # Setup Portfolios
        # self.eigenportfolio = pd.DataFrame(data=eigenportfolio_1, columns=['Investment Weight'], index=self.tickers)
        # self.eigenportfolio2 = pd.DataFrame(data=eigenportfolio_2, columns=['Investment Weight'], index=self.tickers)

        # Construct minimum variance weights
        ones = np.ones(len(inv_cov_mat))
        inv_dot_ones = np.dot(inv_cov_mat, ones)
        min_var_weights = inv_dot_ones / np.dot(inv_dot_ones, ones)

        self.min_var_portfolio = pd.DataFrame(data=min_var_weights, columns=['Investment Weight'], index=self.tickers)

        # Largest eigenvalue eigenportfolio
        D, S = np.linalg.eigh(self.covariance_matrix_in_sample)
        eigenportfolio_1 = S[:, -1] / np.sum(S[:, -1])  # Normalize to sum to 1
        self.eigenportfolio_largest = pd.DataFrame(data=eigenportfolio_1, columns=['Investment Weight'], index=self.tickers)

        # Variance = w^T Sigma w
        largest_var = np.dot(eigenportfolio_1, np.dot(self.covariance_matrix_in_sample, eigenportfolio_1))
        min_var = np.dot(min_var_weights, np.dot(self.covariance_matrix_in_sample, min_var_weights))

        print('Varianza del eigenportfolio: {0} , Varianza del portafolio de minima Varianza: {1}'.format(largest_var,
                                                                                                          min_var))

    def ploting_eigenportfolio(self):
        """
        Each bar indicates the weight associated with a stock.
        These two portfolios are assumed to have uncorrelated returns
        :return: plot
        """
        f = plt.figure()
        ax = plt.subplot(121)
        self.min_var_portfolio.plot(kind='bar', ax=ax, legend=False)
        plt.title("Min Var Portfolio")
        ax = plt.subplot(122)
        self.eigenportfolio_largest.plot(kind='bar', ax=ax, legend=False)
        plt.title("Max E.V. Eigenportfolio")
        print(self.min_var_portfolio)

    def get_cumulative_returns_over_time(self, sample, weights):
        return (((1 + sample).cumprod(axis=0)) - 1).dot(weights)

    def samples(self):
        # In Sample vs. Out of Sample
        """"
        We are really interested in our performance in the test set, but we are glad to know a little
        on performance in the training set.
        """
        self.in_sample_ind = np.arange(0, (self.returns.shape[0] - self.training_period + 1))
        self.out_sample_ind = np.arange((self.returns.shape[0] - self.training_period + 1), self.returns.shape[0])

        self.cumulative_returns = self.get_cumulative_returns_over_time(self.returns, self.min_var_portfolio).values
        self.cumulative_returns_largest = self.get_cumulative_returns_over_time(self.returns,
                                                                                self.eigenportfolio_largest).values

    def ploting_samples(self):

        f = plt.figure(figsize=(10, 4))

        ax = plt.subplot(121)
        ax.plot(self.cumulative_returns[self.in_sample_ind], 'black')
        ax.plot(self.out_sample_ind, self.cumulative_returns[self.out_sample_ind], 'r')
        plt.title("Minimum Variance Portfolio")

        ax = plt.subplot(122)
        ax.plot(self.cumulative_returns_largest[self.in_sample_ind], 'black')
        ax.plot(self.out_sample_ind, self.cumulative_returns_largest[self.out_sample_ind], 'r')
        plt.title("Eigenportfolio")
        plt.show()




