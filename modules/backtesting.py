import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class BacktestEngine:
    def __init__(self):
        self.results = {}
        self.trades = []
        self.equity_curve = pd.Series()

    def run_backtest(self, historical_data: pd.DataFrame, initial_capital: float = 10000,
                    slippage: float = 0.001, position_size: float = 0.1, 
                    fees: Dict[str, float] = None, latency_ms: float = 100,
                    min_spread: float = 0.5, max_risk: int = 7) -> Dict:
        """Run comprehensive backtest on historical data."""
        try:
            if historical_data.empty:
                return {'error': 'No historical data provided'}
            
            if fees is None:
                fees = {'binance': 0.1, 'coinbase': 0.5, 'kraken': 0.25}
            
            # Initialize portfolio
            portfolio = {
                'cash': initial_capital,
                'positions': {},
                'total_value': initial_capital,
                'trades': [],
                'daily_returns': [],
                'drawdowns': []
            }
            
            # Convert historical data to opportunities
            opportunities = self._create_opportunities_from_data(
                historical_data, fees, min_spread, max_risk
            )
            
            if not opportunities:
                return {'error': 'No trading opportunities found in historical data'}
            
            logger.info(f"Running backtest with {len(opportunities)} opportunities")
            
            # Process each opportunity
            equity_values = []
            dates = []
            
            for opportunity in opportunities:
                trade_result = self._execute_backtest_trade(
                    opportunity, portfolio, slippage, position_size, latency_ms
                )
                
                if trade_result:
                    portfolio['trades'].append(trade_result)
                    portfolio['total_value'] = portfolio['cash']
                    
                    # Record equity curve
                    equity_values.append(portfolio['total_value'])
                    dates.append(opportunity['timestamp'])
            
            # Calculate performance metrics
            results = self._calculate_performance_metrics(portfolio, initial_capital, dates, equity_values)
            
            # Store equity curve
            if dates and equity_values:
                self.equity_curve = pd.Series(equity_values, index=dates)
                results['equity_curve'] = self.equity_curve
            
            self.trades = portfolio['trades']
            results['trades'] = self.trades
            
            return results
            
        except Exception as e:
            logger.error(f"Error running backtest: {str(e)}")
            return {'error': str(e)}

    def _create_opportunities_from_data(self, historical_data: pd.DataFrame, 
                                      fees: Dict[str, float], min_spread: float,
                                      max_risk: int) -> List[Dict]:
        """Create arbitrage opportunities from historical price data."""
        opportunities = []
        
        try:
            # Group by timestamp to find simultaneous prices
            grouped = historical_data.groupby('timestamp')
            
            for timestamp, group in grouped:
                if len(group) < 2:
                    continue
                
                # Create price dictionary for this timestamp
                crypto_prices = {}
                for _, row in group.iterrows():
                    crypto = row['crypto']
                    exchange = row['exchange']
                    price = row['price']
                    
                    if crypto not in crypto_prices:
                        crypto_prices[crypto] = {}
                    crypto_prices[crypto][exchange] = price
                
                # Find arbitrage opportunities
                for crypto, exchanges in crypto_prices.items():
                    if len(exchanges) < 2:
                        continue
                    
                    exchange_list = list(exchanges.keys())
                    for i, buy_exchange in enumerate(exchange_list):
                        for j, sell_exchange in enumerate(exchange_list):
                            if i >= j:
                                continue
                            
                            buy_price = exchanges[buy_exchange]
                            sell_price = exchanges[sell_exchange]
                            
                            spread_pct = ((sell_price - buy_price) / buy_price) * 100
                            
                            if spread_pct >= min_spread:
                                # Calculate basic opportunity metrics
                                trade_amount = 1000  # Default trade amount
                                gross_profit = (sell_price - buy_price) * (trade_amount / buy_price)
                                
                                buy_fee = fees.get(buy_exchange, 0.1) / 100
                                sell_fee = fees.get(sell_exchange, 0.1) / 100
                                total_fees = trade_amount * (buy_fee + sell_fee)
                                
                                net_profit = gross_profit - total_fees
                                
                                if net_profit > 0:
                                    # Simple risk score calculation
                                    risk_score = min(10, max(1, int(spread_pct * 2)))
                                    
                                    if risk_score <= max_risk:
                                        opportunity = {
                                            'timestamp': timestamp,
                                            'crypto': crypto,
                                            'buy_exchange': buy_exchange,
                                            'sell_exchange': sell_exchange,
                                            'buy_price': buy_price,
                                            'sell_price': sell_price,
                                            'spread_pct': spread_pct,
                                            'gross_profit': gross_profit,
                                            'net_profit': net_profit,
                                            'risk_score': risk_score,
                                            'trade_amount': trade_amount
                                        }
                                        opportunities.append(opportunity)
            
            # Sort by timestamp
            opportunities.sort(key=lambda x: x['timestamp'])
            
            return opportunities
            
        except Exception as e:
            logger.error(f"Error creating opportunities: {str(e)}")
            return []

    def _execute_backtest_trade(self, opportunity: Dict, portfolio: Dict,
                              slippage: float, position_size: float, 
                              latency_ms: float) -> Optional[Dict]:
        """Execute a single trade in the backtest."""
        try:
            # Calculate position size
            available_cash = portfolio['cash']
            trade_amount = min(
                opportunity['trade_amount'],
                available_cash * position_size
            )
            
            if trade_amount < 100:  # Minimum trade size
                return None
            
            # Apply slippage
            slippage_cost = trade_amount * slippage
            
            # Calculate execution delay impact (simplified)
            latency_impact = min(0.001, latency_ms / 100000)  # Max 0.1% impact
            latency_cost = trade_amount * latency_impact
            
            # Total transaction costs
            total_costs = slippage_cost + latency_cost
            
            # Calculate actual profit after all costs
            base_profit = opportunity['net_profit'] * (trade_amount / opportunity['trade_amount'])
            actual_profit = base_profit - total_costs
            
            # Check if trade is still profitable
            if actual_profit <= 0:
                return None
            
            # Execute trade
            portfolio['cash'] -= total_costs
            portfolio['cash'] += actual_profit
            
            # Record trade
            trade_record = {
                'timestamp': opportunity['timestamp'],
                'crypto': opportunity['crypto'],
                'buy_exchange': opportunity['buy_exchange'],
                'sell_exchange': opportunity['sell_exchange'],
                'buy_price': opportunity['buy_price'],
                'sell_price': opportunity['sell_price'],
                'amount': trade_amount,
                'gross_profit': base_profit,
                'net_profit': actual_profit,
                'slippage_cost': slippage_cost,
                'latency_cost': latency_cost,
                'total_costs': total_costs,
                'success': True
            }
            
            return trade_record
            
        except Exception as e:
            logger.error(f"Error executing backtest trade: {str(e)}")
            return None

    def _calculate_performance_metrics(self, portfolio: Dict, initial_capital: float,
                                     dates: List, equity_values: List) -> Dict:
        """Calculate comprehensive performance metrics."""
        try:
            trades = portfolio['trades']
            final_value = portfolio['total_value']
            
            if not trades:
                return {
                    'total_return': 0.0,
                    'total_trades': 0,
                    'winning_trades': 0,
                    'losing_trades': 0,
                    'win_rate': 0.0,
                    'avg_profit_per_trade': 0.0,
                    'max_drawdown': 0.0,
                    'sharpe_ratio': 0.0,
                    'profit_factor': 0.0
                }
            
            # Basic metrics
            total_return = (final_value - initial_capital) / initial_capital
            total_trades = len(trades)
            
            # Trade analysis
            profits = [trade['net_profit'] for trade in trades]
            winning_trades = len([p for p in profits if p > 0])
            losing_trades = len([p for p in profits if p < 0])
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            avg_profit_per_trade = np.mean(profits) if profits else 0
            
            # Risk metrics
            if len(equity_values) > 1:
                returns = pd.Series(equity_values).pct_change().dropna()
                
                # Maximum drawdown
                peak = pd.Series(equity_values).expanding().max()
                drawdown = (pd.Series(equity_values) - peak) / peak
                max_drawdown = abs(drawdown.min())
                
                # Sharpe ratio (assuming risk-free rate of 2%)
                risk_free_rate = 0.02 / 252  # Daily risk-free rate
                excess_returns = returns - risk_free_rate
                sharpe_ratio = excess_returns.mean() / excess_returns.std() * np.sqrt(252) if excess_returns.std() != 0 else 0
                
                # Volatility
                volatility = returns.std() * np.sqrt(252)
            else:
                max_drawdown = 0.0
                sharpe_ratio = 0.0
                volatility = 0.0
            
            # Profit factor
            gross_profit = sum([p for p in profits if p > 0])
            gross_loss = abs(sum([p for p in profits if p < 0]))
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf') if gross_profit > 0 else 0
            
            # Additional metrics
            if trades:
                trade_durations = []  # Simplified - assume all trades are instantaneous
                avg_trade_duration = 0  # Could be enhanced with actual duration calculation
                
                max_consecutive_wins = self._calculate_consecutive_wins(profits)
                max_consecutive_losses = self._calculate_consecutive_losses(profits)
            else:
                max_consecutive_wins = 0
                max_consecutive_losses = 0
            
            return {
                'total_return': total_return,
                'final_portfolio_value': final_value,
                'total_profit': final_value - initial_capital,
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'losing_trades': losing_trades,
                'win_rate': win_rate,
                'avg_profit_per_trade': avg_profit_per_trade,
                'max_profit': max(profits) if profits else 0,
                'max_loss': min(profits) if profits else 0,
                'max_drawdown': max_drawdown,
                'sharpe_ratio': sharpe_ratio,
                'profit_factor': profit_factor,
                'max_consecutive_wins': max_consecutive_wins,
                'max_consecutive_losses': max_consecutive_losses,
                'gross_profit': gross_profit,
                'gross_loss': gross_loss,
                'volatility': volatility
            }
            
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {str(e)}")
            return {'error': str(e)}

    def _calculate_consecutive_wins(self, profits: List[float]) -> int:
        """Calculate maximum consecutive winning trades."""
        max_consecutive = 0
        current_consecutive = 0
        
        for profit in profits:
            if profit > 0:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0
        
        return max_consecutive

    def _calculate_consecutive_losses(self, profits: List[float]) -> int:
        """Calculate maximum consecutive losing trades."""
        max_consecutive = 0
        current_consecutive = 0
        
        for profit in profits:
            if profit < 0:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0
        
        return max_consecutive

    def analyze_trade_timing(self, trades: List[Dict]) -> Dict:
        """Analyze trade timing patterns."""
        try:
            if not trades:
                return {}
            
            df = pd.DataFrame(trades)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['hour'] = df['timestamp'].dt.hour
            df['day_of_week'] = df['timestamp'].dt.dayofweek
            
            # Hourly analysis
            hourly_profits = df.groupby('hour')['net_profit'].agg(['sum', 'mean', 'count'])
            best_hour = hourly_profits['mean'].idxmax()
            worst_hour = hourly_profits['mean'].idxmin()
            
            # Daily analysis
            daily_profits = df.groupby('day_of_week')['net_profit'].agg(['sum', 'mean', 'count'])
            best_day = daily_profits['mean'].idxmax()
            worst_day = daily_profits['mean'].idxmin()
            
            return {
                'best_trading_hour': int(best_hour),
                'worst_trading_hour': int(worst_hour),
                'best_trading_day': int(best_day),
                'worst_trading_day': int(worst_day),
                'hourly_analysis': hourly_profits.to_dict(),
                'daily_analysis': daily_profits.to_dict()
            }
            
        except Exception as e:
            logger.error(f"Error analyzing trade timing: {str(e)}")
            return {}

    def compare_strategies(self, strategies: Dict[str, Dict]) -> Dict:
        """Compare multiple trading strategies."""
        try:
            comparison = {}
            
            for strategy_name, strategy_params in strategies.items():
                # Run backtest with strategy parameters
                results = self.run_backtest(**strategy_params)
                comparison[strategy_name] = {
                    'total_return': results.get('total_return', 0),
                    'sharpe_ratio': results.get('sharpe_ratio', 0),
                    'max_drawdown': results.get('max_drawdown', 0),
                    'win_rate': results.get('win_rate', 0),
                    'total_trades': results.get('total_trades', 0)
                }
            
            return comparison
            
        except Exception as e:
            logger.error(f"Error comparing strategies: {str(e)}")
            return {}

    def monte_carlo_simulation(self, base_results: Dict, num_simulations: int = 1000) -> Dict:
        """Run Monte Carlo simulation based on historical trade results."""
        try:
            if not self.trades:
                return {'error': 'No trade history for simulation'}
            
            # Extract trade returns
            trade_returns = [trade['net_profit'] for trade in self.trades]
            
            if not trade_returns:
                return {'error': 'No profitable trades for simulation'}
            
            # Run simulations
            final_values = []
            max_drawdowns = []
            
            initial_capital = 10000
            
            for _ in range(num_simulations):
                # Randomly sample trades with replacement
                simulated_trades = np.random.choice(trade_returns, len(trade_returns), replace=True)
                
                # Calculate equity curve
                equity = initial_capital
                peak = initial_capital
                max_dd = 0
                
                for trade_return in simulated_trades:
                    equity += trade_return
                    peak = max(peak, equity)
                    drawdown = (peak - equity) / peak
                    max_dd = max(max_dd, drawdown)
                
                final_values.append(equity)
                max_drawdowns.append(max_dd)
            
            # Calculate statistics
            final_returns = [(fv - initial_capital) / initial_capital for fv in final_values]
            
            return {
                'mean_return': np.mean(final_returns),
                'std_return': np.std(final_returns),
                'percentile_5': np.percentile(final_returns, 5),
                'percentile_95': np.percentile(final_returns, 95),
                'probability_of_loss': len([r for r in final_returns if r < 0]) / len(final_returns),
                'mean_max_drawdown': np.mean(max_drawdowns),
                'worst_case_drawdown': np.max(max_drawdowns)
            }
            
        except Exception as e:
            logger.error(f"Error in Monte Carlo simulation: {str(e)}")
            return {'error': str(e)}
