import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class ArbitrageEngine:
    def __init__(self):
        self.min_liquidity_score = 0.5
        self.risk_weights = {
            'spread': 0.4,
            'volume': 0.3,
            'volatility': 0.2,
            'exchange_reliability': 0.1
        }
        
        # Exchange reliability scores (based on historical uptime and reliability)
        self.exchange_reliability = {
            'coingecko': 0.95,
            'coinpaprika': 0.90,
            'cryptocompare': 0.85,
            'coinlore': 0.80,
            'nomics': 0.75
        }

    def calculate_spread(self, buy_price: float, sell_price: float) -> float:
        """Calculate spread percentage between two prices."""
        if buy_price <= 0 or sell_price <= 0:
            return 0.0
        return ((sell_price - buy_price) / buy_price) * 100

    def calculate_transaction_costs(self, amount: float, buy_fee: float, sell_fee: float) -> float:
        """Calculate total transaction costs."""
        buy_cost = amount * (buy_fee / 100)
        sell_cost = amount * (sell_fee / 100)
        return buy_cost + sell_cost

    def calculate_net_profit(self, gross_profit: float, transaction_costs: float) -> float:
        """Calculate net profit after fees."""
        return gross_profit - transaction_costs

    def calculate_opportunity_score(self, spread_pct: float, net_profit: float, 
                                  buy_exchange: str, sell_exchange: str) -> float:
        """Calculate a composite opportunity score."""
        # Base score from spread percentage
        spread_score = min(spread_pct / 5.0, 1.0)  # Normalize to max 5% spread
        
        # Profit score (normalized)
        profit_score = min(abs(net_profit) / 1000, 1.0)  # Normalize to $1000 max
        
        # Exchange reliability score
        buy_reliability = self.exchange_reliability.get(buy_exchange, 0.5)
        sell_reliability = self.exchange_reliability.get(sell_exchange, 0.5)
        reliability_score = (buy_reliability + sell_reliability) / 2
        
        # Composite score
        opportunity_score = (
            spread_score * self.risk_weights['spread'] +
            profit_score * self.risk_weights['volume'] +
            reliability_score * self.risk_weights['exchange_reliability'] +
            (1 - abs(spread_pct) / 10) * self.risk_weights['volatility']  # Lower volatility is better
        ) * 10  # Scale to 0-10
        
        return max(0, min(10, opportunity_score))

    def calculate_risk_score(self, spread_pct: float, buy_exchange: str, 
                           sell_exchange: str, volatility: float = None) -> int:
        """Calculate risk score (1-10, where 10 is highest risk)."""
        risk_score = 1
        
        # High spread = higher risk (might be stale data)
        if spread_pct > 3.0:
            risk_score += 3
        elif spread_pct > 1.5:
            risk_score += 2
        elif spread_pct > 0.5:
            risk_score += 1
        
        # Exchange reliability factor
        avg_reliability = (
            self.exchange_reliability.get(buy_exchange, 0.5) +
            self.exchange_reliability.get(sell_exchange, 0.5)
        ) / 2
        
        if avg_reliability < 0.7:
            risk_score += 2
        elif avg_reliability < 0.85:
            risk_score += 1
        
        # Volatility factor (if available)
        if volatility:
            if volatility > 0.05:  # High volatility
                risk_score += 2
            elif volatility > 0.03:
                risk_score += 1
        
        return min(10, risk_score)

    def find_opportunities(self, price_data: Dict[str, Dict[str, float]], 
                         fees: Dict[str, float], min_spread: float = 0.5,
                         trade_amount: float = 1000) -> List[Dict]:
        """Find arbitrage opportunities from price data."""
        opportunities = []
        
        for crypto, exchanges in price_data.items():
            if len(exchanges) < 2:
                continue
            
            # Get all exchange pairs
            exchange_list = list(exchanges.keys())
            
            for i, buy_exchange in enumerate(exchange_list):
                for j, sell_exchange in enumerate(exchange_list):
                    if i >= j:  # Avoid duplicate pairs and same exchange
                        continue
                    
                    buy_price = exchanges.get(buy_exchange)
                    sell_price = exchanges.get(sell_exchange)
                    
                    if not buy_price or not sell_price:
                        continue
                    
                    # Calculate spread
                    spread_pct = self.calculate_spread(buy_price, sell_price)
                    
                    # Check if opportunity exists (positive spread above minimum)
                    if spread_pct >= min_spread:
                        # Calculate profits
                        gross_profit = (sell_price - buy_price) * (trade_amount / buy_price)
                        
                        # Calculate transaction costs
                        buy_fee = fees.get(buy_exchange, 0.1)
                        sell_fee = fees.get(sell_exchange, 0.1)
                        transaction_costs = self.calculate_transaction_costs(
                            trade_amount, buy_fee, sell_fee
                        )
                        
                        net_profit = self.calculate_net_profit(gross_profit, transaction_costs)
                        
                        # Only consider profitable opportunities
                        if net_profit > 0:
                            opportunity_score = self.calculate_opportunity_score(
                                spread_pct, net_profit, buy_exchange, sell_exchange
                            )
                            
                            risk_score = self.calculate_risk_score(
                                spread_pct, buy_exchange, sell_exchange
                            )
                            
                            opportunity = {
                                'timestamp': datetime.now(),
                                'crypto': crypto,
                                'buy_exchange': buy_exchange,
                                'sell_exchange': sell_exchange,
                                'buy_price': buy_price,
                                'sell_price': sell_price,
                                'spread_pct': spread_pct,
                                'trade_amount': trade_amount,
                                'gross_profit': gross_profit,
                                'transaction_costs': transaction_costs,
                                'net_profit': net_profit,
                                'opportunity_score': opportunity_score,
                                'risk_score': risk_score,
                                'buy_fee_pct': buy_fee,
                                'sell_fee_pct': sell_fee
                            }
                            
                            opportunities.append(opportunity)
                            
                            logger.info(
                                f"Found opportunity: {crypto} - "
                                f"Buy {buy_exchange} ${buy_price:.2f}, "
                                f"Sell {sell_exchange} ${sell_price:.2f}, "
                                f"Spread: {spread_pct:.2f}%, "
                                f"Net Profit: ${net_profit:.2f}"
                            )
        
        # Sort by opportunity score (descending)
        opportunities.sort(key=lambda x: x['opportunity_score'], reverse=True)
        
        return opportunities

    def filter_opportunities(self, opportunities: List[Dict], 
                           max_risk: int = 7, min_profit: float = 10,
                           min_score: float = 5.0) -> List[Dict]:
        """Filter opportunities based on criteria."""
        filtered = []
        
        for opp in opportunities:
            if (opp['risk_score'] <= max_risk and 
                opp['net_profit'] >= min_profit and
                opp['opportunity_score'] >= min_score):
                filtered.append(opp)
        
        return filtered

    def calculate_position_size(self, available_capital: float, risk_pct: float, 
                              opportunity: Dict) -> float:
        """Calculate optimal position size based on risk management."""
        # Risk-based position sizing
        risk_adjusted_capital = available_capital * (risk_pct / 100)
        
        # Adjust based on opportunity score
        score_multiplier = opportunity['opportunity_score'] / 10
        
        # Adjust based on risk score (inverse relationship)
        risk_multiplier = (11 - opportunity['risk_score']) / 10
        
        position_size = risk_adjusted_capital * score_multiplier * risk_multiplier
        
        # Ensure minimum and maximum position sizes
        min_position = 100  # $100 minimum
        max_position = available_capital * 0.2  # Maximum 20% of capital
        
        return max(min_position, min(max_position, position_size))

    def estimate_execution_time(self, opportunity: Dict) -> float:
        """Estimate execution time in seconds."""
        base_time = 2.0  # Base execution time in seconds
        
        # Add time based on exchange reliability
        buy_reliability = self.exchange_reliability.get(opportunity['buy_exchange'], 0.5)
        sell_reliability = self.exchange_reliability.get(opportunity['sell_exchange'], 0.5)
        
        # Lower reliability = longer execution time
        reliability_factor = 2 - (buy_reliability + sell_reliability) / 2
        
        return base_time * reliability_factor

    def calculate_break_even_spread(self, fees: Dict[str, float], 
                                  buy_exchange: str, sell_exchange: str) -> float:
        """Calculate minimum spread needed to break even."""
        buy_fee = fees.get(buy_exchange, 0.1)
        sell_fee = fees.get(sell_exchange, 0.1)
        
        # Total fee percentage
        total_fees = buy_fee + sell_fee
        
        # Add small buffer for other costs (slippage, etc.)
        break_even_spread = total_fees + 0.1
        
        return break_even_spread

    def analyze_market_conditions(self, price_data: Dict[str, Dict[str, float]]) -> Dict:
        """Analyze current market conditions."""
        analysis = {
            'total_pairs': 0,
            'active_exchanges': set(),
            'price_ranges': {},
            'avg_spreads': {},
            'market_efficiency': 0.0
        }
        
        total_spreads = []
        
        for crypto, exchanges in price_data.items():
            if len(exchanges) < 2:
                continue
            
            analysis['total_pairs'] += 1
            analysis['active_exchanges'].update(exchanges.keys())
            
            prices = list(exchanges.values())
            analysis['price_ranges'][crypto] = {
                'min': min(prices),
                'max': max(prices),
                'range_pct': ((max(prices) - min(prices)) / min(prices)) * 100
            }
            
            # Calculate average spread for this crypto
            spreads = []
            exchange_list = list(exchanges.keys())
            for i, ex1 in enumerate(exchange_list):
                for j, ex2 in enumerate(exchange_list):
                    if i < j:
                        spread = abs(exchanges[ex1] - exchanges[ex2]) / exchanges[ex1] * 100
                        spreads.append(spread)
            
            if spreads:
                analysis['avg_spreads'][crypto] = np.mean(spreads)
                total_spreads.extend(spreads)
        
        # Calculate market efficiency (lower spreads = more efficient)
        if total_spreads:
            avg_spread = np.mean(total_spreads)
            analysis['market_efficiency'] = max(0, 100 - (avg_spread * 20))  # Scale to 0-100
        
        analysis['active_exchanges'] = list(analysis['active_exchanges'])
        
        return analysis
