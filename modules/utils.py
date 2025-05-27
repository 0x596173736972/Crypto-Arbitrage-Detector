import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union
import logging
import re

logger = logging.getLogger(__name__)

def format_currency(amount: float, currency: str = "USD", decimals: int = 2) -> str:
    """Format a number as currency."""
    try:
        if pd.isna(amount) or amount is None:
            return f"${0:.{decimals}f}"
        
        if currency.upper() == "USD":
            if abs(amount) >= 1e9:
                return f"${amount/1e9:.1f}B"
            elif abs(amount) >= 1e6:
                return f"${amount/1e6:.1f}M"
            elif abs(amount) >= 1e3:
                return f"${amount/1e3:.1f}K"
            else:
                return f"${amount:.{decimals}f}"
        else:
            return f"{amount:.{decimals}f} {currency}"
    except (ValueError, TypeError):
        return f"${0:.{decimals}f}"

def format_percentage(value: float, decimals: int = 2) -> str:
    """Format a number as percentage."""
    try:
        if pd.isna(value) or value is None:
            return f"{0:.{decimals}f}%"
        return f"{value:.{decimals}f}%"
    except (ValueError, TypeError):
        return f"{0:.{decimals}f}%"

def format_number(value: float, decimals: int = 2, use_thousands_sep: bool = True) -> str:
    """Format a number with optional thousands separator."""
    try:
        if pd.isna(value) or value is None:
            return f"{0:.{decimals}f}"
        
        if use_thousands_sep:
            return f"{value:,.{decimals}f}"
        else:
            return f"{value:.{decimals}f}"
    except (ValueError, TypeError):
        return f"{0:.{decimals}f}"

def calculate_profit_metrics(opportunities: List[Dict]) -> Dict:
    """Calculate comprehensive profit metrics from opportunities."""
    try:
        if not opportunities:
            return {
                'total_opportunities': 0,
                'total_gross_profit': 0.0,
                'total_net_profit': 0.0,
                'avg_spread': 0.0,
                'avg_profit_per_opportunity': 0.0,
                'best_opportunity': None,
                'profit_by_crypto': {},
                'profit_by_exchange_pair': {}
            }
        
        df = pd.DataFrame(opportunities)
        
        # Basic metrics
        total_opportunities = len(opportunities)
        total_gross_profit = df['gross_profit'].sum()
        total_net_profit = df['net_profit'].sum()
        avg_spread = df['spread_pct'].mean()
        avg_profit_per_opportunity = total_net_profit / total_opportunities
        
        # Best opportunity
        best_opportunity = df.loc[df['net_profit'].idxmax()].to_dict()
        
        # Profit by cryptocurrency
        profit_by_crypto = df.groupby('crypto')['net_profit'].agg(['sum', 'mean', 'count']).to_dict()
        
        # Profit by exchange pair
        df['exchange_pair'] = df['buy_exchange'] + ' -> ' + df['sell_exchange']
        profit_by_exchange_pair = df.groupby('exchange_pair')['net_profit'].agg(['sum', 'mean', 'count']).to_dict()
        
        return {
            'total_opportunities': total_opportunities,
            'total_gross_profit': total_gross_profit,
            'total_net_profit': total_net_profit,
            'avg_spread': avg_spread,
            'avg_profit_per_opportunity': avg_profit_per_opportunity,
            'best_opportunity': best_opportunity,
            'profit_by_crypto': profit_by_crypto,
            'profit_by_exchange_pair': profit_by_exchange_pair
        }
        
    except Exception as e:
        logger.error(f"Error calculating profit metrics: {str(e)}")
        return {
            'total_opportunities': 0,
            'total_gross_profit': 0.0,
            'total_net_profit': 0.0,
            'avg_spread': 0.0,
            'avg_profit_per_opportunity': 0.0,
            'best_opportunity': None,
            'profit_by_crypto': {},
            'profit_by_exchange_pair': {}
        }

def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
    """Calculate Sharpe ratio for a series of returns."""
    try:
        if len(returns) == 0 or returns.std() == 0:
            return 0.0
        
        excess_returns = returns - (risk_free_rate / 252)  # Daily risk-free rate
        return (excess_returns.mean() / excess_returns.std()) * np.sqrt(252)
        
    except Exception as e:
        logger.error(f"Error calculating Sharpe ratio: {str(e)}")
        return 0.0

def calculate_max_drawdown(equity_curve: pd.Series) -> float:
    """Calculate maximum drawdown from equity curve."""
    try:
        if len(equity_curve) == 0:
            return 0.0
        
        peak = equity_curve.expanding().max()
        drawdown = (equity_curve - peak) / peak
        return abs(drawdown.min())
        
    except Exception as e:
        logger.error(f"Error calculating max drawdown: {str(e)}")
        return 0.0

def calculate_volatility(returns: pd.Series, annualize: bool = True) -> float:
    """Calculate volatility (standard deviation) of returns."""
    try:
        if len(returns) == 0:
            return 0.0
        
        volatility = returns.std()
        
        if annualize:
            volatility *= np.sqrt(252)  # Annualize assuming 252 trading days
        
        return volatility
        
    except Exception as e:
        logger.error(f"Error calculating volatility: {str(e)}")
        return 0.0

def calculate_information_ratio(returns: pd.Series, benchmark_returns: pd.Series = None) -> float:
    """Calculate information ratio."""
    try:
        if benchmark_returns is None:
            benchmark_returns = pd.Series([0] * len(returns))
        
        excess_returns = returns - benchmark_returns
        
        if excess_returns.std() == 0:
            return 0.0
        
        return excess_returns.mean() / excess_returns.std()
        
    except Exception as e:
        logger.error(f"Error calculating information ratio: {str(e)}")
        return 0.0

def calculate_calmar_ratio(returns: pd.Series) -> float:
    """Calculate Calmar ratio (annual return / max drawdown)."""
    try:
        if len(returns) == 0:
            return 0.0
        
        annual_return = returns.mean() * 252
        equity_curve = (1 + returns).cumprod()
        max_dd = calculate_max_drawdown(equity_curve)
        
        if max_dd == 0:
            return float('inf') if annual_return > 0 else 0.0
        
        return annual_return / max_dd
        
    except Exception as e:
        logger.error(f"Error calculating Calmar ratio: {str(e)}")
        return 0.0

def validate_price_data(price_data: Dict) -> Tuple[bool, List[str]]:
    """Validate price data structure and values."""
    errors = []
    
    try:
        if not isinstance(price_data, dict):
            errors.append("Price data must be a dictionary")
            return False, errors
        
        for crypto, exchanges in price_data.items():
            if not isinstance(exchanges, dict):
                errors.append(f"Exchange data for {crypto} must be a dictionary")
                continue
            
            for exchange, price in exchanges.items():
                if not isinstance(price, (int, float)):
                    errors.append(f"Price for {crypto} on {exchange} must be a number")
                elif price <= 0:
                    errors.append(f"Price for {crypto} on {exchange} must be positive")
                elif price > 1000000:  # Sanity check
                    errors.append(f"Price for {crypto} on {exchange} seems too high: ${price}")
        
        return len(errors) == 0, errors
        
    except Exception as e:
        errors.append(f"Error validating price data: {str(e)}")
        return False, errors

def clean_currency_string(currency_str: str) -> float:
    """Clean and convert currency string to float."""
    try:
        if not isinstance(currency_str, str):
            return float(currency_str)
        
        # Remove currency symbols and whitespace
        cleaned = re.sub(r'[$€£¥,\s]', '', currency_str)
        
        # Handle percentage
        if '%' in cleaned:
            cleaned = cleaned.replace('%', '')
            return float(cleaned) / 100
        
        return float(cleaned)
        
    except (ValueError, TypeError):
        return 0.0

def time_ago(timestamp: datetime) -> str:
    """Convert timestamp to human-readable time ago format."""
    try:
        if not isinstance(timestamp, datetime):
            timestamp = pd.to_datetime(timestamp)
        
        now = datetime.now()
        diff = now - timestamp
        
        if diff.days > 0:
            return f"{diff.days} day{'s' if diff.days != 1 else ''} ago"
        elif diff.seconds > 3600:
            hours = diff.seconds // 3600
            return f"{hours} hour{'s' if hours != 1 else ''} ago"
        elif diff.seconds > 60:
            minutes = diff.seconds // 60
            return f"{minutes} minute{'s' if minutes != 1 else ''} ago"
        else:
            return "Just now"
            
    except Exception as e:
        logger.error(f"Error formatting time ago: {str(e)}")
        return "Unknown"

def calculate_compound_annual_growth_rate(start_value: float, end_value: float, years: float) -> float:
    """Calculate Compound Annual Growth Rate (CAGR)."""
    try:
        if start_value <= 0 or end_value <= 0 or years <= 0:
            return 0.0
        
        return (end_value / start_value) ** (1 / years) - 1
        
    except Exception as e:
        logger.error(f"Error calculating CAGR: {str(e)}")
        return 0.0

def detect_outliers_zscore(data: pd.Series, threshold: float = 3.0) -> pd.Series:
    """Detect outliers using Z-score method."""
    try:
        z_scores = np.abs((data - data.mean()) / data.std())
        return z_scores > threshold
        
    except Exception as e:
        logger.error(f"Error detecting outliers: {str(e)}")
        return pd.Series([False] * len(data))

def normalize_data(data: pd.Series, method: str = 'minmax') -> pd.Series:
    """Normalize data using specified method."""
    try:
        if method == 'minmax':
            return (data - data.min()) / (data.max() - data.min())
        elif method == 'zscore':
            return (data - data.mean()) / data.std()
        elif method == 'robust':
            median = data.median()
            mad = (data - median).abs().median()
            return (data - median) / mad
        else:
            logger.warning(f"Unknown normalization method: {method}")
            return data
            
    except Exception as e:
        logger.error(f"Error normalizing data: {str(e)}")
        return data

def calculate_correlation_significance(correlation: float, n_samples: int, alpha: float = 0.05) -> Tuple[bool, float]:
    """Test if correlation is statistically significant."""
    try:
        if n_samples <= 2:
            return False, 1.0
        
        # Calculate t-statistic
        t_stat = correlation * np.sqrt((n_samples - 2) / (1 - correlation**2))
        
        # Calculate degrees of freedom
        df = n_samples - 2
        
        # Critical t-value (two-tailed test)
        from scipy import stats
        critical_t = stats.t.ppf(1 - alpha/2, df)
        
        # P-value
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df))
        
        is_significant = abs(t_stat) > critical_t
        
        return is_significant, p_value
        
    except Exception as e:
        logger.error(f"Error testing correlation significance: {str(e)}")
        return False, 1.0

def create_risk_score(spread_pct: float, volatility: float, exchange_reliability: float) -> int:
    """Create a risk score from 1-10 based on multiple factors."""
    try:
        risk_score = 1
        
        # Spread component (higher spread can indicate stale data or high risk)
        if spread_pct > 5.0:
            risk_score += 4
        elif spread_pct > 2.0:
            risk_score += 3
        elif spread_pct > 1.0:
            risk_score += 2
        elif spread_pct > 0.5:
            risk_score += 1
        
        # Volatility component
        if volatility > 0.1:  # 10% volatility
            risk_score += 3
        elif volatility > 0.05:  # 5% volatility
            risk_score += 2
        elif volatility > 0.02:  # 2% volatility
            risk_score += 1
        
        # Exchange reliability component (inverse relationship)
        if exchange_reliability < 0.7:
            risk_score += 2
        elif exchange_reliability < 0.85:
            risk_score += 1
        
        return min(10, max(1, risk_score))
        
    except Exception as e:
        logger.error(f"Error creating risk score: {str(e)}")
        return 5  # Default medium risk

def calculate_position_size_kelly(win_rate: float, avg_win: float, avg_loss: float, 
                                capital: float, max_position_pct: float = 0.25) -> float:
    """Calculate optimal position size using Kelly Criterion."""
    try:
        if avg_loss <= 0 or win_rate <= 0 or win_rate >= 1:
            return capital * 0.01  # Conservative 1% if parameters are invalid
        
        # Kelly fraction: f = (bp - q) / b
        # where b = avg_win/avg_loss, p = win_rate, q = loss_rate
        b = avg_win / abs(avg_loss)
        p = win_rate
        q = 1 - win_rate
        
        kelly_fraction = (b * p - q) / b
        
        # Cap the Kelly fraction to avoid over-leveraging
        kelly_fraction = max(0, min(kelly_fraction, max_position_pct))
        
        return capital * kelly_fraction
        
    except Exception as e:
        logger.error(f"Error calculating Kelly position size: {str(e)}")
        return capital * 0.01

def generate_trade_id(crypto: str, buy_exchange: str, sell_exchange: str, timestamp: datetime = None) -> str:
    """Generate a unique trade ID."""
    try:
        if timestamp is None:
            timestamp = datetime.now()
        
        timestamp_str = timestamp.strftime("%Y%m%d_%H%M%S")
        return f"{crypto}_{buy_exchange}_{sell_exchange}_{timestamp_str}"
        
    except Exception as e:
        logger.error(f"Error generating trade ID: {str(e)}")
        return f"TRADE_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

def calculate_slippage_impact(trade_size: float, avg_daily_volume: float, 
                            market_impact_factor: float = 0.1) -> float:
    """Estimate slippage impact based on trade size and market liquidity."""
    try:
        if avg_daily_volume <= 0:
            return 0.01  # Default 1% slippage if no volume data
        
        # Simple square root market impact model
        volume_ratio = trade_size / avg_daily_volume
        slippage = market_impact_factor * np.sqrt(volume_ratio)
        
        # Cap slippage at reasonable levels
        return min(slippage, 0.05)  # Max 5% slippage
        
    except Exception as e:
        logger.error(f"Error calculating slippage impact: {str(e)}")
        return 0.01

def format_execution_time(milliseconds: float) -> str:
    """Format execution time in human-readable format."""
    try:
        if milliseconds < 1000:
            return f"{milliseconds:.0f}ms"
        else:
            seconds = milliseconds / 1000
            return f"{seconds:.1f}s"
            
    except Exception as e:
        logger.error(f"Error formatting execution time: {str(e)}")
        return "N/A"

def calculate_opportunity_quality_score(spread_pct: float, volume_score: float, 
                                      reliability_score: float, time_score: float) -> float:
    """Calculate a composite opportunity quality score."""
    try:
        # Weighted average of different factors
        weights = {
            'spread': 0.4,
            'volume': 0.3,
            'reliability': 0.2,
            'timing': 0.1
        }
        
        # Normalize spread score (higher spread = better opportunity up to a point)
        spread_normalized = min(spread_pct / 2.0, 1.0)  # Cap at 2% for normalization
        
        quality_score = (
            spread_normalized * weights['spread'] +
            volume_score * weights['volume'] +
            reliability_score * weights['reliability'] +
            time_score * weights['timing']
        ) * 10  # Scale to 0-10
        
        return max(0, min(10, quality_score))
        
    except Exception as e:
        logger.error(f"Error calculating opportunity quality score: {str(e)}")
        return 5.0

def export_to_csv(data: Union[pd.DataFrame, List[Dict]], filename: str = None) -> str:
    """Export data to CSV format and return as string."""
    try:
        if isinstance(data, list):
            df = pd.DataFrame(data)
        else:
            df = data.copy()
        
        # Format numeric columns
        for col in df.select_dtypes(include=[np.number]).columns:
            if 'pct' in col.lower() or 'percent' in col.lower():
                df[col] = df[col].round(2)
            elif 'price' in col.lower() or 'profit' in col.lower():
                df[col] = df[col].round(4)
        
        return df.to_csv(index=False)
        
    except Exception as e:
        logger.error(f"Error exporting to CSV: {str(e)}")
        return ""

def sanitize_filename(filename: str) -> str:
    """Sanitize filename by removing invalid characters."""
    try:
        # Remove invalid characters
        sanitized = re.sub(r'[<>:"/\\|?*]', '_', filename)
        
        # Remove multiple underscores
        sanitized = re.sub(r'_+', '_', sanitized)
        
        # Remove leading/trailing underscores
        sanitized = sanitized.strip('_')
        
        return sanitized
        
    except Exception as e:
        logger.error(f"Error sanitizing filename: {str(e)}")
        return "export_file"

def calculate_portfolio_metrics(trades: List[Dict], initial_capital: float = 10000) -> Dict:
    """Calculate comprehensive portfolio performance metrics."""
    try:
        if not trades:
            return {
                'total_return': 0.0,
                'annualized_return': 0.0,
                'volatility': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'calmar_ratio': 0.0,
                'win_rate': 0.0,
                'profit_factor': 0.0
            }
        
        df = pd.DataFrame(trades)
        
        # Calculate returns
        returns = df['net_profit'] / initial_capital
        cumulative_returns = (1 + returns).cumprod()
        
        # Time period
        start_date = pd.to_datetime(df['timestamp'].iloc[0])
        end_date = pd.to_datetime(df['timestamp'].iloc[-1])
        days = (end_date - start_date).days
        years = max(days / 365.25, 1/365.25)  # Minimum 1 day
        
        # Metrics
        total_return = cumulative_returns.iloc[-1] - 1
        annualized_return = calculate_compound_annual_growth_rate(1, cumulative_returns.iloc[-1], years)
        volatility = calculate_volatility(returns)
        sharpe_ratio = calculate_sharpe_ratio(returns)
        max_drawdown = calculate_max_drawdown(cumulative_returns)
        calmar_ratio = calculate_calmar_ratio(returns)
        
        # Trade metrics
        winning_trades = len(df[df['net_profit'] > 0])
        total_trades = len(df)
        win_rate = winning_trades / total_trades
        
        gross_profit = df[df['net_profit'] > 0]['net_profit'].sum()
        gross_loss = abs(df[df['net_profit'] < 0]['net_profit'].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar_ratio,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'gross_profit': gross_profit,
            'gross_loss': gross_loss
        }
        
    except Exception as e:
        logger.error(f"Error calculating portfolio metrics: {str(e)}")
        return {}
