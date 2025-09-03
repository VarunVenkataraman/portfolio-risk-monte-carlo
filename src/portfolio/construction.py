"""
Portfolio construction and management module.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class Portfolio:
    """
    Portfolio construction and management class.
    """
    
    def __init__(self, 
                 tickers: List[str],
                 weights: Optional[Dict[str, float]] = None,
                 notional_value: float = 100000.0):
        """
        Initialize portfolio with tickers and weights.
        
        Args:
            tickers: List of asset ticker symbols
            weights: Dictionary of asset weights (equal weight if None)
            notional_value: Total portfolio value in dollars
        """
        self.tickers = tickers
        self.notional_value = notional_value
        
        if weights is None:
            # Equal weighting
            weight_per_asset = 1.0 / len(tickers)
            self.weights = {ticker: weight_per_asset for ticker in tickers}
        else:
            self.weights = weights
            
        self._validate_weights()
        
    def _validate_weights(self):
        """Validate that weights sum to 1.0 and match tickers."""
        if abs(sum(self.weights.values()) - 1.0) > 1e-6:
            raise ValueError(f"Weights sum to {sum(self.weights.values()):.6f}, not 1.0")
            
        if set(self.weights.keys()) != set(self.tickers):
            raise ValueError("Weight keys must match tickers exactly")
    
    def get_weights_array(self) -> np.ndarray:
        """Get weights as numpy array in ticker order."""
        return np.array([self.weights[ticker] for ticker in self.tickers])
    
    def calculate_portfolio_returns(self, returns: pd.DataFrame) -> pd.Series:
        """
        Calculate portfolio returns from asset returns.
        
        Args:
            returns: DataFrame with asset returns
            
        Returns:
            Series with portfolio returns
        """
        weights_array = self.get_weights_array()
        portfolio_returns = (returns[self.tickers] * weights_array).sum(axis=1)
        return portfolio_returns
    
    def calculate_portfolio_values(self, returns: pd.DataFrame) -> pd.Series:
        """
        Calculate portfolio values over time.
        
        Args:
            returns: DataFrame with asset returns
            
        Returns:
            Series with portfolio values
        """
        portfolio_returns = self.calculate_portfolio_returns(returns)
        portfolio_values = self.notional_value * (1 + portfolio_returns).cumprod()
        return portfolio_values
    
    def get_allocation_details(self) -> Dict[str, float]:
        """Get portfolio allocation in dollar amounts."""
        return {
            ticker: weight * self.notional_value 
            for ticker, weight in self.weights.items()
        }
    
    def rebalance_weights(self, new_weights: Dict[str, float]):
        """
        Rebalance portfolio with new weights.
        
        Args:
            new_weights: Dictionary of new asset weights
        """
        self.weights = new_weights
        self._validate_weights()
        logger.info("Portfolio rebalanced with new weights")


class PortfolioOptimizer:
    """
    Portfolio optimization utilities.
    """
    
    @staticmethod
    def calculate_portfolio_stats(returns: pd.DataFrame, 
                                 weights: np.ndarray) -> Dict[str, float]:
        """
        Calculate portfolio statistics.
        
        Args:
            returns: DataFrame with asset returns
            weights: Array of portfolio weights
            
        Returns:
            Dictionary with portfolio statistics
        """
        portfolio_returns = (returns * weights).sum(axis=1)
        
        stats = {
            'annual_return': portfolio_returns.mean() * 252,
            'annual_volatility': portfolio_returns.std() * np.sqrt(252),
            'sharpe_ratio': (portfolio_returns.mean() * 252) / (portfolio_returns.std() * np.sqrt(252)),
            'max_drawdown': PortfolioOptimizer._calculate_max_drawdown(portfolio_returns),
            'skewness': portfolio_returns.skew(),
            'kurtosis': portfolio_returns.kurtosis()
        }
        
        return stats
    
    @staticmethod
    def _calculate_max_drawdown(returns: pd.Series) -> float:
        """Calculate maximum drawdown from returns series."""
        cumulative = (1 + returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative - rolling_max) / rolling_max
        return drawdown.min()
    
    @staticmethod
    def calculate_correlation_matrix(returns: pd.DataFrame) -> pd.DataFrame:
        """Calculate correlation matrix of asset returns."""
        return returns.corr()
    
    @staticmethod
    def calculate_covariance_matrix(returns: pd.DataFrame, 
                                   annualized: bool = True) -> pd.DataFrame:
        """
        Calculate covariance matrix of asset returns.
        
        Args:
            returns: DataFrame with asset returns
            annualized: Whether to annualize the covariance matrix
            
        Returns:
            Covariance matrix
        """
        cov_matrix = returns.cov()
        if annualized:
            cov_matrix *= 252  # Assume daily returns
        return cov_matrix
    
    @staticmethod
    def risk_parity_weights(returns: pd.DataFrame) -> np.ndarray:
        """
        Calculate risk parity weights (equal risk contribution).
        
        Args:
            returns: DataFrame with asset returns
            
        Returns:
            Array of risk parity weights
        """
        cov_matrix = returns.cov().values
        n_assets = len(returns.columns)
        
        # Simple risk parity approximation: inverse volatility weighting
        vol = returns.std().values
        inv_vol_weights = (1 / vol) / (1 / vol).sum()
        
        return inv_vol_weights
    
    @staticmethod
    def minimum_variance_weights(returns: pd.DataFrame) -> np.ndarray:
        """
        Calculate minimum variance portfolio weights.
        
        Args:
            returns: DataFrame with asset returns
            
        Returns:
            Array of minimum variance weights
        """
        cov_matrix = returns.cov().values
        n_assets = len(returns.columns)
        
        # Minimum variance: w = inv(Cov) * 1 / (1' * inv(Cov) * 1)
        inv_cov = np.linalg.inv(cov_matrix)
        ones = np.ones((n_assets, 1))
        weights = (inv_cov @ ones) / (ones.T @ inv_cov @ ones)
        
        return weights.flatten()


class PortfolioAnalyzer:
    """
    Portfolio analysis and reporting utilities.
    """
    
    def __init__(self, portfolio: Portfolio):
        """Initialize analyzer with a portfolio."""
        self.portfolio = portfolio
    
    def generate_performance_report(self, 
                                   returns: pd.DataFrame) -> Dict[str, any]:
        """
        Generate comprehensive performance report.
        
        Args:
            returns: DataFrame with asset returns
            
        Returns:
            Dictionary with performance metrics
        """
        portfolio_returns = self.portfolio.calculate_portfolio_returns(returns)
        portfolio_values = self.portfolio.calculate_portfolio_values(returns)
        
        report = {
            'portfolio_stats': PortfolioOptimizer.calculate_portfolio_stats(
                returns[self.portfolio.tickers], 
                self.portfolio.get_weights_array()
            ),
            'asset_stats': {},
            'correlation_matrix': PortfolioOptimizer.calculate_correlation_matrix(
                returns[self.portfolio.tickers]
            ),
            'weights': self.portfolio.weights,
            'allocation_dollars': self.portfolio.get_allocation_details(),
            'final_value': portfolio_values.iloc[-1],
            'total_return': (portfolio_values.iloc[-1] / self.portfolio.notional_value) - 1
        }
        
        # Individual asset statistics
        for ticker in self.portfolio.tickers:
            asset_returns = returns[ticker]
            report['asset_stats'][ticker] = {
                'annual_return': asset_returns.mean() * 252,
                'annual_volatility': asset_returns.std() * np.sqrt(252),
                'sharpe_ratio': (asset_returns.mean() * 252) / (asset_returns.std() * np.sqrt(252)),
                'weight': self.portfolio.weights[ticker]
            }
        
        return report