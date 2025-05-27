import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.tsa.stattools import adfuller, coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen
import warnings
from typing import Dict, List, Tuple, Optional
import logging

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

class StatisticalAnalyzer:
    def __init__(self):
        self.window_size = 20  # Default rolling window size
        self.confidence_level = 0.05  # 95% confidence level

    def calculate_rolling_zscore(self, series: pd.Series, window: int = None) -> pd.Series:
        """Calculate rolling Z-score for a time series."""
        if window is None:
            window = self.window_size
        
        try:
            rolling_mean = series.rolling(window=window).mean()
            rolling_std = series.rolling(window=window).std()
            
            # Avoid division by zero
            rolling_std = rolling_std.replace(0, np.nan)
            z_score = (series - rolling_mean) / rolling_std
            
            return z_score.fillna(0)
            
        except Exception as e:
            logger.error(f"Error calculating Z-score: {str(e)}")
            return pd.Series([0] * len(series), index=series.index)

    def augmented_dickey_fuller_test(self, series: pd.Series) -> Dict:
        """Perform Augmented Dickey-Fuller test for stationarity."""
        try:
            # Remove NaN values
            clean_series = series.dropna()
            
            if len(clean_series) < 10:
                return {
                    'statistic': np.nan,
                    'pvalue': np.nan,
                    'critical_values': {},
                    'is_stationary': False,
                    'error': 'Insufficient data points'
                }
            
            result = adfuller(clean_series, autolag='AIC')
            
            return {
                'statistic': result[0],
                'pvalue': result[1],
                'critical_values': result[4],
                'is_stationary': result[1] < self.confidence_level,
                'interpretation': 'Stationary' if result[1] < self.confidence_level else 'Non-stationary'
            }
            
        except Exception as e:
            logger.error(f"Error in ADF test: {str(e)}")
            return {
                'statistic': np.nan,
                'pvalue': np.nan,
                'critical_values': {},
                'is_stationary': False,
                'error': str(e)
            }

    def engle_granger_cointegration(self, price_series1: pd.Series, 
                                  price_series2: pd.Series) -> Dict:
        """Perform Engle-Granger cointegration test."""
        try:
            # Align series by index
            aligned_data = pd.concat([price_series1, price_series2], axis=1).dropna()
            
            if len(aligned_data) < 20:
                return {
                    'statistic': np.nan,
                    'pvalue': np.nan,
                    'critical_values': [],
                    'is_cointegrated': False,
                    'error': 'Insufficient aligned data points'
                }
            
            series1 = aligned_data.iloc[:, 0]
            series2 = aligned_data.iloc[:, 1]
            
            # Perform cointegration test
            result = coint(series1, series2)
            
            return {
                'statistic': result[0],
                'pvalue': result[1],
                'critical_values': result[2],
                'is_cointegrated': result[1] < self.confidence_level,
                'interpretation': 'Cointegrated' if result[1] < self.confidence_level else 'Not cointegrated'
            }
            
        except Exception as e:
            logger.error(f"Error in Engle-Granger test: {str(e)}")
            return {
                'statistic': np.nan,
                'pvalue': np.nan,
                'critical_values': [],
                'is_cointegrated': False,
                'error': str(e)
            }

    def johansen_cointegration(self, price_matrix: pd.DataFrame) -> Dict:
        """Perform Johansen cointegration test for multiple series."""
        try:
            # Remove NaN values
            clean_data = price_matrix.dropna()
            
            if len(clean_data) < 20 or clean_data.shape[1] < 2:
                return {
                    'trace_stats': [],
                    'eigen_stats': [],
                    'critical_values': [],
                    'cointegration_rank': 0,
                    'error': 'Insufficient data or variables'
                }
            
            # Perform Johansen test
            result = coint_johansen(clean_data, det_order=0, k_ar_diff=1)
            
            # Determine cointegration rank
            trace_stats = result.lr1
            critical_values_trace = result.cvt
            
            cointegration_rank = 0
            for i in range(len(trace_stats)):
                if trace_stats[i] > critical_values_trace[i, 1]:  # 5% significance level
                    cointegration_rank += 1
            
            return {
                'trace_stats': trace_stats.tolist(),
                'eigen_stats': result.lr2.tolist(),
                'critical_values_trace': critical_values_trace.tolist(),
                'critical_values_eigen': result.cvm.tolist(),
                'cointegration_rank': cointegration_rank,
                'eigenvectors': result.evec.tolist()
            }
            
        except Exception as e:
            logger.error(f"Error in Johansen test: {str(e)}")
            return {
                'trace_stats': [],
                'eigen_stats': [],
                'critical_values': [],
                'cointegration_rank': 0,
                'error': str(e)
            }

    def calculate_correlation_matrix(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """Calculate correlation matrix for multiple price series."""
        try:
            return price_data.corr()
        except Exception as e:
            logger.error(f"Error calculating correlation matrix: {str(e)}")
            return pd.DataFrame()

    def calculate_volatility(self, price_series: pd.Series, window: int = None) -> pd.Series:
        """Calculate rolling volatility (standard deviation of returns)."""
        if window is None:
            window = self.window_size
        
        try:
            # Calculate returns
            returns = price_series.pct_change().dropna()
            
            # Calculate rolling volatility
            volatility = returns.rolling(window=window).std() * np.sqrt(window)
            
            return volatility.fillna(0)
            
        except Exception as e:
            logger.error(f"Error calculating volatility: {str(e)}")
            return pd.Series([0] * len(price_series), index=price_series.index)

    def detect_outliers_iqr(self, series: pd.Series, multiplier: float = 1.5) -> pd.Series:
        """Detect outliers using Interquartile Range method."""
        try:
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - multiplier * IQR
            upper_bound = Q3 + multiplier * IQR
            
            outliers = (series < lower_bound) | (series > upper_bound)
            return outliers
            
        except Exception as e:
            logger.error(f"Error detecting outliers: {str(e)}")
            return pd.Series([False] * len(series), index=series.index)

    def calculate_moving_averages(self, series: pd.Series, 
                                windows: List[int] = [5, 10, 20, 50]) -> pd.DataFrame:
        """Calculate multiple moving averages."""
        try:
            ma_data = pd.DataFrame(index=series.index)
            ma_data['price'] = series
            
            for window in windows:
                ma_data[f'MA_{window}'] = series.rolling(window=window).mean()
            
            return ma_data
            
        except Exception as e:
            logger.error(f"Error calculating moving averages: {str(e)}")
            return pd.DataFrame()

    def calculate_bollinger_bands(self, series: pd.Series, window: int = 20, 
                                num_std: float = 2) -> pd.DataFrame:
        """Calculate Bollinger Bands."""
        try:
            ma = series.rolling(window=window).mean()
            std = series.rolling(window=window).std()
            
            upper_band = ma + (std * num_std)
            lower_band = ma - (std * num_std)
            
            bands = pd.DataFrame({
                'price': series,
                'middle_band': ma,
                'upper_band': upper_band,
                'lower_band': lower_band
            })
            
            # Calculate position relative to bands
            bands['bb_position'] = (series - lower_band) / (upper_band - lower_band)
            
            return bands
            
        except Exception as e:
            logger.error(f"Error calculating Bollinger Bands: {str(e)}")
            return pd.DataFrame()

    def calculate_rsi(self, series: pd.Series, window: int = 14) -> pd.Series:
        """Calculate Relative Strength Index."""
        try:
            delta = series.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
            
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            return rsi.fillna(50)  # Neutral RSI for NaN values
            
        except Exception as e:
            logger.error(f"Error calculating RSI: {str(e)}")
            return pd.Series([50] * len(series), index=series.index)

    def analyze_spreads(self, spread_data: pd.DataFrame) -> Dict:
        """Comprehensive analysis of spread data."""
        try:
            if spread_data.empty or 'spread_pct' not in spread_data.columns:
                return {'error': 'No spread data available'}
            
            spreads = spread_data['spread_pct'].dropna()
            
            if len(spreads) < 10:
                return {'error': 'Insufficient spread data for analysis'}
            
            # Basic statistics
            analysis = {
                'count': len(spreads),
                'mean_spread': spreads.mean(),
                'median_spread': spreads.median(),
                'std_spread': spreads.std(),
                'min_spread': spreads.min(),
                'max_spread': spreads.max(),
                'skewness': stats.skew(spreads),
                'kurtosis': stats.kurtosis(spreads)
            }
            
            # Stationarity test
            adf_result = self.augmented_dickey_fuller_test(spreads)
            analysis.update({
                'adf_statistic': adf_result['statistic'],
                'adf_pvalue': adf_result['pvalue'],
                'is_stationary': adf_result['is_stationary']
            })
            
            # Calculate Z-scores
            z_scores = self.calculate_rolling_zscore(spreads)
            analysis.update({
                'z_score_mean': z_scores.mean(),
                'z_score_std': z_scores.std(),
                'extreme_z_scores': len(z_scores[abs(z_scores) > 2])
            })
            
            # Volatility analysis
            volatility = self.calculate_volatility(spreads)
            analysis.update({
                'avg_volatility': volatility.mean(),
                'volatility_trend': 'increasing' if volatility.iloc[-5:].mean() > volatility.iloc[-15:-5].mean() else 'decreasing'
            })
            
            # Outlier detection
            outliers = self.detect_outliers_iqr(spreads)
            analysis['outlier_count'] = outliers.sum()
            analysis['outlier_percentage'] = (outliers.sum() / len(spreads)) * 100
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error in spread analysis: {str(e)}")
            return {'error': str(e)}

    def calculate_half_life(self, spread_series: pd.Series) -> float:
        """Calculate mean reversion half-life."""
        try:
            # Calculate lagged series
            spread_lag = spread_series.shift(1).dropna()
            spread_diff = spread_series.diff().dropna()
            
            # Align series
            aligned_data = pd.concat([spread_diff, spread_lag], axis=1).dropna()
            
            if len(aligned_data) < 10:
                return np.nan
            
            # Perform regression: ΔY(t) = α + βY(t-1) + ε(t)
            y = aligned_data.iloc[:, 0]  # spread differences
            x = aligned_data.iloc[:, 1]  # lagged spreads
            
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            
            # Calculate half-life
            if slope < 0:
                half_life = -np.log(2) / slope
                return half_life
            else:
                return np.inf  # No mean reversion
                
        except Exception as e:
            logger.error(f"Error calculating half-life: {str(e)}")
            return np.nan

    def perform_normality_test(self, series: pd.Series) -> Dict:
        """Perform Shapiro-Wilk normality test."""
        try:
            clean_series = series.dropna()
            
            if len(clean_series) < 3:
                return {
                    'statistic': np.nan,
                    'pvalue': np.nan,
                    'is_normal': False,
                    'error': 'Insufficient data'
                }
            
            # Use Shapiro-Wilk test for smaller samples, Kolmogorov-Smirnov for larger
            if len(clean_series) <= 5000:
                statistic, pvalue = stats.shapiro(clean_series)
                test_name = 'Shapiro-Wilk'
            else:
                statistic, pvalue = stats.kstest(clean_series, 'norm')
                test_name = 'Kolmogorov-Smirnov'
            
            return {
                'test_name': test_name,
                'statistic': statistic,
                'pvalue': pvalue,
                'is_normal': pvalue > self.confidence_level
            }
            
        except Exception as e:
            logger.error(f"Error in normality test: {str(e)}")
            return {
                'statistic': np.nan,
                'pvalue': np.nan,
                'is_normal': False,
                'error': str(e)
            }

    def calculate_information_ratio(self, spread_series: pd.Series, 
                                  benchmark_return: float = 0) -> float:
        """Calculate information ratio for spread trading."""
        try:
            returns = spread_series.pct_change().dropna()
            excess_returns = returns - benchmark_return
            
            if len(excess_returns) == 0:
                return 0.0
            
            return excess_returns.mean() / excess_returns.std() if excess_returns.std() != 0 else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating information ratio: {str(e)}")
            return 0.0
