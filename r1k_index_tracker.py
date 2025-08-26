"""
Russell 1000 Index Replication

This script implements several methods to replicate the Russell 1000 index
using its top 25 constituents:
1. Market Cap Weighting
2. Linear Regression
3. Ridge Regression with CV
4. Tracking Error Minimization
5. Factor-Based Optimization
6. Iteratively Reweighted L1 (IRL1) for sparse tracking

Author: Tinatin Nikvashvili
Date: May 9, 2025
"""

import os
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from functools import partial
from operator import itemgetter
import yfinance as yf
import pandas_market_calendars as mcal
from pandas_datareader import data as pdr
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import TimeSeriesSplit
import cvxpy as cp
import math
import pdb
# Suppress warnings
warnings.filterwarnings('ignore')

# Create output directory for plots if it doesn't exist
PLOT_DIR = 'plots'
os.makedirs(PLOT_DIR, exist_ok=True)

# Constants - could move to config file in production
# top 25 largest constituents of r1K as of march 31, 2024 - I ended up pulling data from a
# paid vendor so hard-coded for now, for future improvement I could also pull this automatically
# could add this as constants to config file for cleaner code but submission instructions ask for 1 script
TOP_25_R1K_TICKERS = [
    'MSFT', 'AAPL', 'NVDA', 'GOOG', 'AMZN', 'META', 'BRK-B', 'LLY', 'JPM', 
    'V', 'TSLA', 'AVGO', 'WMT', 'XOM', 'UNH', 'MA', 'HD', 'PG', 'JNJ', 
    'ORCL', 'MRK', 'COST', 'ABBV', 'BAC', 'CVX'
]
R1K_TICKER = ['^RUI']
TRAIN_START = '2023-01-01'
TRAIN_END = '2024-03-31'
TEST_START = '2024-04-01'
TEST_END = '2024-05-01'  # yfinance end date is exclusive



class RussellIndexReplicator:
    """Class to handle Russell 1000 index replication experiment"""
    
    def __init__(self, r1k_ticker, constituent_tickers, 
                 train_start, train_end, test_start, test_end):
        """Initialize with date ranges and tickers"""
        self.r1k_ticker = r1k_ticker
        self.constituent_tickers = constituent_tickers
        self.train_start = train_start
        self.train_end = train_end
        self.test_start = test_start
        self.test_end = test_end
        
        # Data containers
        self.prices = None
        self.r1k = None
        self.top_25 = None
        self.returns = None
        self.train_r1k = None
        self.train_top_25 = None
        self.test_r1k = None
        self.test_top_25 = None
        self.train_r1k_ret = None
        self.train_25_ret = None
        self.test_r1k_ret = None
        self.test_25_ret = None
        self.factors = None
        
        # Weight containers for different methods
        self.weights = {}
        
    def download_data(self):
        """Download price data and align to trading calendar"""
        print("Downloading historical prices...")
        
        # Download Russell 1000 index data
        self.r1k = self.download_close(self.r1k_ticker, self.train_start, self.test_end)
        self.r1k, _ = self.align_prices_to_nyse(self.r1k)
        print(f"Missing values for r1k before filling: {self.r1k.isna().sum().sum()}")
        self.r1k = self.ffill_if_missing(self.r1k)
        
        # Download top 25 constituents data
        self.top_25 = self.download_close(self.constituent_tickers, self.train_start, self.test_end)
        self.top_25, _ = self.align_prices_to_nyse(self.top_25)
        print(f"Missing values for top25 before filling: {self.top_25.isna().sum().sum()}")
        self.top_25 = self.ffill_if_missing(self.top_25)
        
        # Combine all price data
        self.prices = pd.concat([self.top_25, self.r1k], axis=1)
        print(f"Original data shape: {self.prices.shape}")
        
        # Calculate returns
        self.returns = self.compute_returns(self.prices)
        
        # Check for data quality issues
        self.check_data_quality()
    
    def prepare_datasets(self):
        """Split data into training and test sets"""
        print("Preparing training and test datasets...")
        
        # Split price data
        self.train_r1k = self.r1k[self.train_start:self.train_end]
        self.train_top_25 = self.top_25[self.train_start:self.train_end]
        self.test_r1k = self.r1k[self.test_start:self.test_end]
        self.test_top_25 = self.top_25[self.test_start:self.test_end]
        
        # Compute returns for each dataset
        self.train_r1k_ret = self.compute_returns(self.train_r1k)
        self.train_25_ret = self.compute_returns(self.train_top_25)
        self.test_r1k_ret = self.compute_returns(self.test_r1k)
        self.test_25_ret = self.compute_returns(self.test_top_25)
        
        # Print dataset information
        self.print_dataset_info()
        
        # Load factor data for factor-based model
        self.factors = self.load_factors(self.train_start, self.train_end)
        
    def build_models(self):
        """Build and evaluate all replication models"""
        print("Building replication models...")
        
        # 1. Market Cap Weighting (Benchmark)
        print("1. Computing market cap weights...")
        self.weights['market_cap'], market_caps = self.get_market_cap_weights(
            self.constituent_tickers, self.train_end
        )
        
        # 2. Linear Regression
        print("2. Training linear regression model...")
        reg_model = LinearRegression(fit_intercept=True, positive=True)
        reg_model.fit(self.train_25_ret, self.train_r1k_ret)
        self.weights['linear'] = pd.Series(reg_model.coef_[0], index=self.constituent_tickers)
        self.weights['linear'] = self.weights['linear'] / self.weights['linear'].sum()
        
        # 3. Ridge Regression with CV
        print("3. Training ridge regression with CV...")
        self.weights['ridge'] = self.train_ridge_regression()
        
        # 4. Tracking Error Minimization
        print("4. Optimizing for minimum tracking error...")
        self.weights['te_optimized'] = self.minimize_tracking_error()
        
        # 5. Factor-Based Optimization
        print("5. Running factor-based optimization...")
        self.weights['factor_opt'] = self.run_factor_optimization()
        
        # 6. IRL1 Sparse Tracking
        print("6. Running IRL1 sparse tracking...")
        self.weights['sparse'] = self.run_irl1_optimization()
    
    def evaluate_models(self):
        """Evaluate all models with cross-validation"""
        print("Evaluating models with cross-validation...")
        
        fold_tes = {f'w_{k}': [] for k in self.weights.keys()}
        tscv = TimeSeriesSplit(n_splits=3)
        
        for train_idx, val_idx in tscv.split(self.train_25_ret):
            X_tr = self.train_25_ret.iloc[train_idx]
            y_tr = self.train_r1k_ret.iloc[train_idx].values.ravel()
            X_val = self.train_25_ret.iloc[val_idx]
            y_val = self.train_r1k_ret.iloc[val_idx].values.ravel()
            
            for key, weight in self.weights.items():
                fold_tes[f'w_{key}'].append(
                    self.calculate_tracking_error(X_val.dot(weight), y_val)
                )
        
        # Calculate mean TE for each method
        mean_tes = {k: np.mean(v) for k, v in fold_tes.items()}
        
        # Sort by tracking error (lower is better)
        sorted_tes = sorted(mean_tes.items(), key=itemgetter(1))
        
        print("\nCV Tracking Error Results (annualized):")
        for method, te in sorted_tes:
            print(f"{method}: {te:.4%}")
        
        # Select best method for final evaluation
        best_method = sorted_tes[0][0].replace('w_', '')
        self.best_weights = self.weights[best_method]
        print(f"\nBest method: {best_method} with TE: {sorted_tes[0][1]:.4%}")
        
        return best_method
    
    def evaluate_on_test_set(self):
        """Evaluate the best model on the test set"""
        print("Evaluating on test set...")
        
        # Calculate simulated index values during test period
        russell_scaled = self.normalize_series(self.test_r1k['^RUI'])
        bench_prices = self.calculate_prices(self.test_25_ret, self.weights['market_cap'])
        best_prices = self.calculate_prices(self.test_25_ret, self.best_weights)
        
        # Plot results
        self.plot_test_performance(russell_scaled, best_prices, bench_prices)
        
        # Print performance statistics
        russel_ret = russell_scaled.pct_change().dropna()[1:]  # Skip first NaN
        best_ret = best_prices.pct_change().dropna()
        bench_ret = bench_prices.pct_change().dropna()
        
        print('\nBest Replicate Performance:')
        self.compute_performance_stats(best_ret, russel_ret)
        
        print('\nMarket Cap Benchmark Performance:')
        self.compute_performance_stats(bench_ret, russel_ret)

    def visualize_data(self):
        """Generate and save diagnostic visualizations"""
        print("Generating visualizations...")
        
        # 1. Correlation matrix of constituent returns
        self.plot_correlation_matrix()
        
        # 2. Weight distribution comparison
        self.plot_weight_comparison()
        
        # 3. Check for outliers
        self.plot_outliers()
    
    # =============== HELPER METHODS ===============
    
    @staticmethod
    def download_close(tickers, start_date, end_date):
        """Download historical prices for given tickers and date range from Yahoo Finance"""
        try:
            prices_df = yf.download(tickers, start=start_date, end=end_date)
            close = prices_df['Close']
            return close
        except Exception as e:
            print(f"Error downloading data: {e}")
            raise
    
    @staticmethod
    def align_prices_to_nyse(prices, calendar_name="NYSE"):
        """Reindex prices to NYSE trading days"""
        try:
            # Get the calendar
            nyse = mcal.get_calendar(calendar_name)
            
            # Determine the date range
            start = prices.index.min().date()
            end = prices.index.max().date()
            
            # Build the schedule
            schedule = nyse.schedule(start_date=start, end_date=end)
            
            # How many trading days?
            n_days = len(schedule)
            
            # Reindex prices
            reindexed_prices = prices.reindex(schedule.index)
            
            return reindexed_prices, n_days
        except Exception as e:
            print(f"Error aligning prices to NYSE calendar: {e}")
            raise
    
    @staticmethod
    def ffill_if_missing(df):
        """Forward-fill null values in the DataFrame if any are present"""
        if df.isnull().values.any():
            return df.fillna(method='ffill')
        return df
    
    @staticmethod
    def compute_returns(price_df):
        """Calculate daily returns given price data"""
        returns_df = price_df.pct_change().dropna()
        return returns_df
    
    def check_data_quality(self):
        """Check for data quality issues like stale prices and outliers"""
        # Check for negative prices (should never happen)
        assert self.prices.where(self.prices < 0).sum().sum() == 0, "Negative prices detected"
        
        # Check for stale prices (identical values for multiple consecutive days)
        stale_prices = {}
        for column in self.prices.columns:
            # Check for 3+ consecutive identical prices
            diffs = self.prices[column].diff()
            consecutive_zeros = (diffs == 0).astype(int)
            consecutive_count = consecutive_zeros.groupby(
                (consecutive_zeros != consecutive_zeros.shift()).cumsum()
            ).cumsum()
            if consecutive_count.max() >= 3:
                stale_prices[column] = consecutive_count.max()
        
        if stale_prices:
            print("Warning: Stale prices detected:")
            for column, count in stale_prices.items():
                print(f"  {column}: {count} consecutive identical prices")
        
        # Check for outliers
        zscores = (self.returns - self.returns.mean()) / self.returns.std()
        outlier_mask_per_column = (zscores.abs() > 5).any()
        outliers = self.returns[(zscores.abs() > 5)].dropna(axis=1, how='all')
        
        # Identify outliers
        outlier_tickers = self.returns.columns[outlier_mask_per_column]
        if not outlier_tickers.empty:
            print("Outliers detected for:", list(outlier_tickers))
            
            for column in outliers.columns:
                outlier_dates = outliers.index[outliers[column].notnull()]
                if len(outlier_dates) > 0:
                    print(f"Outliers in {column}: {len(outlier_dates)}")
                    for date in outlier_dates:
                        print(f"  {date}: {self.returns.loc[date, column]:.2%}")
        
        # Store outliers for later visualization
        self.outliers = outliers
    
    def print_dataset_info(self):
        """Print dataset information"""
        print('\nR1K Data Train', '------' * 10)
        print(self.train_r1k.index.min(), self.train_r1k.index.max())
        print(self.train_r1k.shape)
        print('Missing Data:', self.train_r1k.isnull().sum().sum())
        
        print('\nTop 25 Data Train', '------' * 10)
        print(self.train_top_25.index.min(), self.train_top_25.index.max())
        print(self.train_top_25.shape)
        print('Missing Data:', self.train_top_25.isnull().sum().sum())
        
        print('\nR1K Data Test', '------' * 10)
        print(self.test_r1k.index.min(), self.test_r1k.index.max())
        print(self.test_r1k.shape)
        print('Missing Data:', self.test_r1k.isnull().sum().sum())
        
        print('\nTop 25 Data Test', '------' * 10)
        print(self.test_top_25.index.min(), self.test_top_25.index.max())
        print(self.test_top_25.shape)
        print('Missing Data:', self.test_top_25.isnull().sum().sum())
        
        print(f'\nR1K train return shape: {self.train_r1k_ret.shape}')
        print(f'Top 25 R1K train Constituents shape: {self.train_25_ret.shape}')
        print(f'R1K test return shape: {self.test_r1k_ret.shape}')
        print(f'Top 25 R1K test Constituents shape: {self.test_25_ret.shape}')
    
    def get_market_cap_weights(self, tickers, date):
        """
        Get market cap weights as of a specified date
        
        Parameters:
        tickers (list): List of ticker symbols
        date (str): Date in YYYY-MM-DD format to get market caps for
        
        Returns:
        pd.Series: Market cap weights indexed by ticker
        pd.Series: Market caps indexed by ticker
        """
        print(f"Getting market cap weights as of {date}...")
        
        # Convert date string to datetime object if necessary
        as_of_date = pd.to_datetime(date)
        
        # Get historical stock data including the target date
        # We'll get data for a few days before to ensure we have the data 
        # in case the as_of_date falls on no-trading days
        start_date = as_of_date - pd.Timedelta(days=5)
        end_date = as_of_date + pd.Timedelta(days=1)
        
        market_caps = {}
        for ticker in tickers:
            max_retries = 3
            retry_count = 0
            success = False
            
            while not success and retry_count < max_retries:
                try:
                    # Get historical price data
                    stock_data = yf.download(ticker, start=start_date, end=end_date)
                    
                    # If date exactly matches, use that date, otherwise use the closest previous date
                    available_dates = stock_data.index
                    closest_date = available_dates[available_dates <= as_of_date].max()
                    
                    # Get shares outstanding information
                    stock_info = yf.Ticker(ticker)
                    shares_outstanding = stock_info.info.get('sharesOutstanding', 0)
                    splits = stock_info.splits
                    
                    # Adjust for any splits that occurred after the as-of date
                    splits_since = splits[splits.index > date]
                    if not splits_since.empty:
                        cum_factor = splits_since.prod()
                        print(f'Adjusting for splits for {ticker} {cum_factor}')
                        shares_outstanding = shares_outstanding / cum_factor
                    
                    # Calculate market cap as price × shares outstanding
                    if not stock_data.empty and shares_outstanding > 0:
                        price = stock_data.loc[closest_date, 'Close']
                        if isinstance(price, pd.Series):
                            price = price.iloc[0]
                        market_caps[ticker] = price * shares_outstanding
                        success = True
                    else:
                        print(f"Warning: Could not get market cap data for {ticker} as of {date}")
                        market_caps[ticker] = np.nan
                        success = True  # Consider this a "success" to avoid retrying when the data is simply missing
                    
                    # Sleep to avoid API rate limits
                    time.sleep(1)
                    
                except Exception as e:
                    retry_count += 1
                    if "Connection reset by peer" in str(e) and retry_count < max_retries:
                        print(f"Connection reset error for {ticker}. Retrying {retry_count}/{max_retries}...")
                        time.sleep(3 * retry_count)  # Exponential backoff
                    else:
                        print(f"Error getting market cap for {ticker}: {e}")
                        market_caps[ticker] = np.nan
                        success = True  # Mark as done to exit the retry loop
        
        # Handle any missing values
        if pd.isna(pd.Series(market_caps)).any():
            print("Warning: Some market caps are missing. Using mean market cap for missing values.")
            missing_tickers = [t for t, mc in market_caps.items() if pd.isna(mc)]
            mean_mc = pd.Series(market_caps).mean(skipna=True)
            for ticker in missing_tickers:
                market_caps[ticker] = mean_mc
        
        # Calculate weights
        market_caps_series = pd.Series(market_caps)
        total_market_cap = market_caps_series.sum()
        weights = market_caps_series / total_market_cap
        
        return weights, market_caps_series
    
    def train_ridge_regression(self):
        """Train ridge regression model with cross-validation"""
        alphas = np.logspace(-4, 2, 30)
        tscv = TimeSeriesSplit(n_splits=3)
        
        best_alpha, best_cv_te = None, np.inf
        for alpha in alphas:
            fold_tes = []
            for train_idx, val_idx in tscv.split(self.train_25_ret):
                X_tr = self.train_25_ret.iloc[train_idx]
                y_tr = self.train_r1k_ret.iloc[train_idx].values.ravel()
                X_val = self.train_25_ret.iloc[val_idx]
                y_val = self.train_r1k_ret.iloc[val_idx].values.ravel()
                
                model = Ridge(alpha=alpha, fit_intercept=True, positive=True)
                model.fit(X_tr, y_tr)
                
                w = pd.Series(model.coef_, index=self.train_25_ret.columns).clip(lower=0)
                w /= w.sum()
                
                port_val = X_val.dot(w)
                te_val = self.calculate_tracking_error(port_val, y_val)
                fold_tes.append(te_val)
            
            cv_te = np.mean(fold_tes)
            if cv_te < best_cv_te:
                best_cv_te, best_alpha = cv_te, alpha
        
        print(f"Ridge regression: Best CV α = {best_alpha:.1e}, CV-TE = {best_cv_te:.4%}")
        
        # Refit on full training data with best alpha
        final_ridge = Ridge(alpha=best_alpha, fit_intercept=True, positive=True)
        final_ridge.fit(self.train_25_ret, self.train_r1k_ret.values.ravel())
        w_ridge = pd.Series(final_ridge.coef_, index=self.train_25_ret.columns).clip(lower=0)
        w_ridge /= w_ridge.sum()
        
        return w_ridge
    
    def minimize_tracking_error(self):
        """Optimize weights to minimize tracking error"""
        def objective_function(weights, returns, index_returns):
            portfolio_returns = returns.dot(weights)
            tracking_error = np.sum((portfolio_returns - index_returns)**2)
            return tracking_error
        
        # Constraints: weights sum to 1 and are non-negative
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1},)
        bounds = tuple((0, 1) for _ in range(len(self.constituent_tickers)))
        initial_weights = np.ones(len(self.constituent_tickers)) / len(self.constituent_tickers)
        
        try:
            optimization_result = minimize(
                objective_function,
                initial_weights,
                args=(self.train_25_ret.values, self.train_r1k_ret.values),
                method='SLSQP',
                bounds=bounds,
                constraints=constraints
            )
            
            if not optimization_result.success:
                print(f"Warning: Optimization did not converge: {optimization_result.message}")
            
            w_te_optimized = pd.Series(optimization_result.x, index=self.constituent_tickers)
            return w_te_optimized
            
        except Exception as e:
            print(f"Error in tracking error minimization: {e}")
            # Fallback to equal weights if optimization fails
            return pd.Series(initial_weights, index=self.constituent_tickers)
    
    def run_factor_optimization(self):
        """Run factor-based optimization"""
        # Align dates
        common_index = (self.train_25_ret.index
                         .intersection(self.train_r1k_ret.index)
                         .intersection(self.factors.index))
        
        R = self.train_25_ret.loc[common_index]
        r_idx = self.train_r1k_ret.loc[common_index]
        F = self.factors.loc[common_index]
        
        # Estimate factor betas
        # Index beta
        beta_idx, *_ = np.linalg.lstsq(F.values, r_idx.values, rcond=None)
        
        # Stock betas
        B = np.column_stack([
            np.linalg.lstsq(F.values, R[col].values, rcond=None)[0]
            for col in R.columns
        ])
        
        # Tune gamma parameter
        gammas = [0.01, 0.1, 1, 10]
        best_factor = self.tune_factor_te(B, beta_idx.ravel(), R.values, r_idx.values.ravel(), gammas)
        best_gamma = best_factor['γ']
        
        w_factor_opt = pd.Series(best_factor['w'], index=self.train_25_ret.columns)
        w_factor_opt /= w_factor_opt.sum()  # normalize just in case
        
        print(f"Factor model: Best γ: {best_gamma}, CV-TE = {best_factor['te']:.4%}")
        
        return w_factor_opt
    
    def tune_factor_te(self, B, beta_idx, R_mat, r_idx_vec, gammas):
        """Tune gamma parameter for factor model"""
        tscv = TimeSeriesSplit(n_splits=3)
        best = {'γ': None, 'te': np.inf, 'w': None}
        
        for γ in gammas:
            tes = []
            for train_idx, val_idx in tscv.split(R_mat):
                R_tr, r_tr = R_mat[train_idx], r_idx_vec[train_idx]
                B_tr, β_tr = B, beta_idx  # factor loadings assumed constant
                
                # Re-optimize on train slice
                cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1},)
                bnds = [(0, 1)] * R_tr.shape[1]
                obj = lambda w: np.linalg.norm(B_tr.dot(w) - β_tr)**2 + γ * np.sum((r_tr - R_tr.dot(w))**2)
                w0 = np.ones(R_tr.shape[1]) / R_tr.shape[1]
                
                try:
                    result = minimize(obj, w0, bounds=bnds, constraints=cons)
                    if result.success:
                        w_opt = result.x
                    else:
                        print(f"Warning: Factor optimization did not converge for γ={γ}")
                        continue
                except Exception as e:
                    print(f"Error in factor optimization for γ={γ}: {e}")
                    continue
                
                # Val TE
                R_val, r_val = R_mat[val_idx], r_idx_vec[val_idx]
                port_val = R_val.dot(w_opt)
                tes.append(np.sqrt(np.mean((port_val - r_val)**2)) * np.sqrt(252))
            
            if tes:  # Only update if we have valid results
                avg_te = np.mean(tes)
                if avg_te < best['te']:
                    best.update({'γ': γ, 'te': avg_te, 'w': w_opt.copy()})
        
        return best
    
    def run_irl1_optimization(self):
        """Run IRL1 sparse tracking optimization"""
        R = self.train_25_ret.values
        r_idx = self.train_r1k_ret.values.ravel()
        
        # Tune lambda parameter
        lambdas = np.logspace(-4, 0, 10)
        best_irl1 = self.tune_irl1(self.irl1_index_tracking, R, r_idx, lambdas)
        
        w_sparse = pd.Series(best_irl1['w'], index=self.train_25_ret.columns)
        print(f"IRL1: Best λ: {best_irl1['λ']}, CV-TE = {best_irl1['te']:.4%}")
        
        return w_sparse
    
    def tune_irl1(self, IRL1_func, R_mat, r_idx_vec, lambdas):
        """Tune lambda parameter for IRL1 model"""
        tscv = TimeSeriesSplit(n_splits=3)
        best = {'λ': None, 'te': np.inf, 'w': None}
        
        for lam in lambdas:
            tes = []
            for train_idx, val_idx in tscv.split(R_mat):
                try:
                    w_tr = IRL1_func(R_mat[train_idx], r_idx_vec[train_idx], lam)
                    
                    # Val TE
                    port_val = R_mat[val_idx].dot(w_tr)
                    tes.append(np.sqrt(np.mean((port_val - r_idx_vec[val_idx])**2)) * np.sqrt(252))
                except Exception as e:
                    print(f"Error in IRL1 for λ={lam}: {e}")
                    continue
            
            if tes:  # Only update if we have valid results
                avg_te = np.mean(tes)
                if avg_te < best['te']:
                    best.update({'λ': lam, 'te': avg_te, 'w': w_tr.copy()})
        
        return best
    
    @staticmethod
    def irl1_index_tracking(R, r_idx, lam, eps=1e-3, max_iter=50, tol=1e-6):
        """
        Iteratively Reweighted ℓ1 for sparse index tracking.
        
        Minimize   ||R w - r_idx||_2^2  +  lam * sum_i |w_i|
        s.t.       sum(w) = 1,  w >= 0
        
        Args:
          R        (T×N array): component returns
          r_idx    (T-vector):  index returns
          lam      float:       sparsity-controlling parameter
          eps      float:       small constant to avoid division by zero
          max_iter int:         max IRL1 iterations
          tol      float:       stopping criterion on ||w_new - w_old||
          
        Returns:
          w        (N-vector):  final weight solution
        """
        T, N = R.shape
        # Precompute Q for the quadratic term
        P = 2 * (R.T @ R)                 #  N×N
        # And the constant part of the linear term
        c0 = -2 * (R.T @ r_idx)           #  N-vector
        
        # Initialize with uniform weights
        w = np.ones(N) / N
        
        for k in range(max_iter):
            # Build the current ℓ1 penalty weights
            u = 1.0 / (np.abs(w) + eps)    #  N-vector

            # Define and solve the QP:
            #   minimize  ½ wᵀP w + (c0 + lam*u)ᵀ w
            #   s.t.      sum(w) == 1,  w >= 0
            w_var = cp.Variable(N)
            u = 1.0/(np.abs(w) + eps)         # from the last IRL1 iterate
            h = c0 + lam*u  
            obj = 0.5 * cp.quad_form(w_var, P) + cp.sum(cp.multiply(h, w_var))
            constraints = [cp.sum(w_var) == 1, w_var >= 0]
            prob = cp.Problem(cp.Minimize(obj), constraints)
            prob.solve(solver=cp.OSQP)

            w_new = w_var.value
            if w_new is None:
                raise RuntimeError("QP failed to solve at iteration %d" % k)

            # Convergence check
            if np.linalg.norm(w_new - w, 2) < tol:
                w = w_new
                break
            w = w_new

        return w
    
    def load_factors(self, start, end):
        """Load Fama-French factors for the given date range"""
        ff = pdr.DataReader('F-F_Research_Data_Factors_daily', 'famafrench', start, end)[0]
        ff = ff/100.0  # convert to decimal returns
        mom = pdr.DataReader('F-F_Momentum_Factor_daily', 'famafrench', start, end)[0]['Mom'] / 100.0
        factors = pd.concat([ff, mom], axis=1).dropna()
        # Add risk-free back to market
        factors['Mkt'] = factors['Mkt-RF'] + factors['RF']/100.0 * 0  # skip RF term
        return factors
    
    def calculate_tracking_error(self, portfolio_returns, index_returns):
        """Calculate annualized tracking error"""
        tracking_diff = portfolio_returns - index_returns
        return np.sqrt(np.mean(tracking_diff**2)) * np.sqrt(252)  # Annualized
    
    def calculate_prices(self, returns, weights):
        """Calculate cumulative returns given weights"""
        portfolio_returns = returns.dot(weights)
        prices = self.normalize_series((1 + portfolio_returns).cumprod())
        return prices
    
    @staticmethod
    def normalize_series(series, base_value=100.0):
        """Normalize a price series so that the first value equals base_value"""
        return series / series.iloc[0] * base_value
    
    def plot_test_performance(self, russell_scaled, best_prices, bench_prices):
        """Plot test period performance comparison"""
        plt.figure(figsize=(10,6))
        plt.plot(russell_scaled.index, russell_scaled.values, label='Russell 1000 (^RUI)')
        
        # Calculate tracking errors for legend, ensuring alignment
        best_ret = best_prices.pct_change().dropna()
        bench_ret = bench_prices.pct_change().dropna()
        russel_ret = russell_scaled.pct_change().dropna()[1:]  # Skip first NaN to match evaluate_on_test_set
        
        # Align indexes
        common_idx = best_ret.index.intersection(russel_ret.index)
        best_ret_aligned = best_ret.loc[common_idx]
        bench_ret_aligned = bench_ret.loc[common_idx] 
        russel_ret_aligned = russel_ret.loc[common_idx]
        
        # Calculate tracking errors using standard deviation (same as in compute_performance_stats)
        best_diff = best_ret_aligned - russel_ret_aligned
        best_te = best_diff.std() * np.sqrt(252)
        
        bench_diff = bench_ret_aligned - russel_ret_aligned
        bench_te = bench_diff.std() * np.sqrt(252)
        
        plt.plot(best_prices.index, best_prices.values, 
                label=f'Optimized Replicate (TE: {best_te:.2%})')
        plt.plot(bench_prices.index, bench_prices.values, 
                label=f'MC-Weighted Benchmark (TE: {bench_te:.2%})')
        
        plt.title('April 2024: Russell 1000 vs. Replicate vs. MC Benchmark')
        plt.xlabel('Date')
        plt.ylabel('Index Level (Base=100)')
        plt.legend(loc='best')  # Place legend inside the plot at the optimal location
        plt.xticks(rotation=45)
        plt.grid(True, linestyle='--', alpha=0.7)  # Add gridlines
        plt.tight_layout()
        
        # Save plot
        plt.savefig(os.path.join(PLOT_DIR, 'test_performance.png'), 
                   bbox_inches='tight', dpi=300)
        plt.close()
    
    def compute_performance_stats(self, ret, russel_ret):
        """Compute and print performance statistics"""
        # Align the indexes if they're different
        if list(ret.index) != list(russel_ret.index):
            common_idx = ret.index.intersection(russel_ret.index)
            ret = ret.loc[common_idx]
            russel_ret = russel_ret.loc[common_idx]
        
        diff = ret - russel_ret
        te = diff.std() * np.sqrt(252)  # annualized tracking error
        ir = diff.mean() / diff.std() * np.sqrt(252)
        r2 = np.corrcoef(ret, russel_ret)[0,1]**2
        print(f"Annualized Tracking Error: {te:.4%}")
        print(f"Information Ratio: {ir:.4f}")
        print(f"R-squared: {r2:.4f}")
    
    def plot_correlation_matrix(self):
        """Plot correlation matrix of constituent returns"""
        corr = self.train_25_ret.corr()
        
        plt.figure(figsize=(8,8))
        im = plt.imshow(corr, cmap='Blues', vmin=-1, vmax=1)
        
        # Add colorbar
        plt.colorbar(im)
        
        # Set up tick positions and labels
        n = len(corr.columns)
        plt.xticks(range(n), corr.columns, rotation=90)
        plt.yticks(range(n), corr.index)
        
        # Move x-ticks to top
        plt.gca().xaxis.set_ticks_position('top')
        plt.gca().xaxis.set_label_position('top')
        
        plt.title('Correlation Matrix of Constituent Returns')
        plt.tight_layout()
        
        # Save plot
        plt.savefig(os.path.join(PLOT_DIR, 'correlation_matrix.png'))
        plt.close()
    
    def plot_weight_comparison(self):
        """Plot weight distribution comparison across methods"""
        # Create DataFrame of weights
        weight_df = pd.DataFrame(self.weights)
        
        # Plot
        plt.figure(figsize=(12,6))
        weight_df.plot(kind='bar', width=0.8)
        plt.title('Weight Distribution Comparison')
        plt.xlabel('Constituent')
        plt.ylabel('Weight')
        plt.xticks(rotation=45)
        plt.legend(title='Method')
        plt.grid(True, linestyle='--', alpha=0.7)  # Add gridlines
        plt.tight_layout()
        
        # Save plot
        plt.savefig(os.path.join(PLOT_DIR, 'weight_comparison.png'))
        plt.close()
    
    def plot_outliers(self):
        """Plot outliers in price data"""
        if not hasattr(self, 'outliers') or self.outliers.empty:
            return
        
        # Get normalized prices
        normalized_prices = self.normalize_series(self.prices)
        
        # Get only tickers with outliers
        outlier_tickers = self.outliers.columns.tolist()
        if not outlier_tickers:
            return
            
        # Parameters for subplot grid
        n = len(outlier_tickers)
        cols = min(5, n)  # Maximum 5 columns
        rows = math.ceil(n / cols)
        
        # Create a grid with independent y-axes
        fig, axes = plt.subplots(rows, cols, sharex=True, sharey=False,
                                figsize=(cols*3, rows*2.5))
        if rows == 1 and cols == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        
        # Plot each ticker in its own subplot with its own y-axis scale
        for i, col in enumerate(outlier_tickers):
            ax = axes[i]
            ax.plot(normalized_prices.index, normalized_prices[col], label=col)
            # Mark outliers
            outlier_dates = self.outliers.index[self.outliers[col].notnull()]
            ax.scatter(outlier_dates,
                      normalized_prices.loc[outlier_dates, col],
                      color='red', s=20, marker='x')
            ax.set_title(col, fontsize=8)
            ax.grid(True, linestyle='--', alpha=0.5)  # Consistent gridlines
            # Rotate x-axis labels for readability
            ax.tick_params(axis='x', rotation=45)
        
        # Turn off unused subplots
        for ax in axes[n:]:
            ax.axis('off')
        
        plt.suptitle('Normalized Prices with Outlier Marks (Individual Y-Axis)', y=1.02)
        plt.tight_layout()
        
        # Save plot
        plt.savefig(os.path.join(PLOT_DIR, 'outliers.png'))
        plt.close()

    def plot_pred_vs_actual(self,
                            predicted: pd.Series,
                            actual: pd.Series,
                            kind: str = "return"):
        """
        Scatterplot of predicted vs. actual index values.

        Parameters
        ----------
        predicted : pd.Series
            Model’s predicted daily returns (or prices), indexed by date.
        actual : pd.Series
            True index daily returns (or prices), indexed by date.
        kind : str
            “return” or “price” – used in labels & filename.
        """
        # align
        idx = predicted.index.intersection(actual.index)
        p = predicted.loc[idx]
        a = actual.loc[idx]

        plt.figure(figsize=(6,6))
        plt.scatter(p, a, alpha=0.7)
        mn, mx = min(p.min(), a.min()), max(p.max(), a.max())
        plt.plot([mn, mx], [mn, mx], 'r--', label='y = x')
        plt.xlabel(f"Predicted {kind.capitalize()}")
        plt.ylabel(f"Actual {kind.capitalize()}")
        plt.title(f"Scatter: Predicted vs Actual {kind.capitalize()}")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        fname = os.path.join(PLOT_DIR, f"scatter_{kind}.png")
        plt.savefig(fname, dpi=300, bbox_inches='tight')
        plt.close()

def main():
    """Main function to run the Russell 1000 index replication experiment"""
    # Initialize replicator
    replicator = RussellIndexReplicator(
        r1k_ticker=R1K_TICKER,
        constituent_tickers=TOP_25_R1K_TICKERS,
        train_start=TRAIN_START,
        train_end=TRAIN_END,
        test_start=TEST_START,
        test_end=TEST_END
    )
    
    # Run the experiment
    replicator.download_data()
    replicator.prepare_datasets()
    replicator.build_models()
    best_method = replicator.evaluate_models()
    replicator.evaluate_on_test_set()
   

    best_ret = replicator.test_25_ret.dot(replicator.best_weights)
    true_ret = replicator.test_r1k_ret.squeeze()   # (T,)
    replicator.plot_pred_vs_actual(best_ret, true_ret, kind="return")

    russell_scaled = replicator.normalize_series(replicator.test_r1k["^RUI"])
    best_price = replicator.calculate_prices(replicator.test_25_ret,
                                             replicator.best_weights)
    replicator.plot_pred_vs_actual(best_price, russell_scaled, kind="price")

    replicator.visualize_data()
    
    print(f"\nExperiment completed. Best method: {best_method}")
    print(f"Results saved in {PLOT_DIR} directory")

if __name__ == "__main__":
    main()