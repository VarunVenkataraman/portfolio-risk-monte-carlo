"""
Risk metrics calculation module for VaR, CVaR, and other risk measures.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from scipy import stats
import logging

logger = logging.getLogger(__name__)


class RiskMetrics:
    """
    Risk metrics calculation and analysis.
    """
    
    @staticmethod
    def value_at_risk(returns: Union[np.ndarray, pd.Series], 
                     confidence_level: float = 0.95,
                     method: str = "historical") -> float:
        """
        Calculate Value at Risk (VaR).
        
        Args:
            returns: Array or Series of returns/P&L
            confidence_level: Confidence level (e.g., 0.95 for 95% VaR)
            method: Calculation method ("historical", "parametric", "monte_carlo")
            
        Returns:
            VaR value (negative indicates loss)
        """
        if method == "historical":
            return np.percentile(returns, (1 - confidence_level) * 100)
        
        elif method == "parametric":
            mean_return = np.mean(returns)
            std_return = np.std(returns, ddof=1)
            z_score = stats.norm.ppf(1 - confidence_level)
            return mean_return + z_score * std_return
        
        elif method == "monte_carlo":
            # For Monte Carlo, returns should already be simulated
            return np.percentile(returns, (1 - confidence_level) * 100)
        
        else:
            raise ValueError(f"Unsupported VaR method: {method}")
    
    @staticmethod
    def conditional_value_at_risk(returns: Union[np.ndarray, pd.Series], 
                                 confidence_level: float = 0.95,
                                 method: str = "historical") -> float:
        """
        Calculate Conditional Value at Risk (CVaR/Expected Shortfall).
        
        Args:
            returns: Array or Series of returns/P&L
            confidence_level: Confidence level
            method: Calculation method
            
        Returns:
            CVaR value (average of tail losses)
        """
        var_threshold = RiskMetrics.value_at_risk(returns, confidence_level, method)
        
        # Calculate average of returns below VaR threshold
        tail_returns = returns[returns <= var_threshold]
        
        if len(tail_returns) == 0:
            logger.warning(f"No returns below VaR threshold {var_threshold}")
            return var_threshold
        
        return np.mean(tail_returns)
    
    @staticmethod
    def calculate_multiple_vars(returns: Union[np.ndarray, pd.Series],
                               confidence_levels: List[float] = [0.90, 0.95, 0.99],
                               methods: List[str] = ["historical"]) -> Dict[str, Dict[str, float]]:
        """
        Calculate VaR and CVaR for multiple confidence levels and methods.
        
        Args:
            returns: Array or Series of returns/P&L
            confidence_levels: List of confidence levels
            methods: List of calculation methods
            
        Returns:
            Dictionary with VaR and CVaR results
        """
        results = {}
        
        for method in methods:
            results[method] = {}
            
            for conf_level in confidence_levels:
                var = RiskMetrics.value_at_risk(returns, conf_level, method)
                cvar = RiskMetrics.conditional_value_at_risk(returns, conf_level, method)
                
                results[method][f'var_{int(conf_level*100)}'] = var
                results[method][f'cvar_{int(conf_level*100)}'] = cvar
        
        return results
    
    @staticmethod
    def maximum_drawdown(prices_or_returns: Union[np.ndarray, pd.Series],
                        is_returns: bool = True) -> Dict[str, float]:
        """
        Calculate maximum drawdown and related statistics.
        
        Args:
            prices_or_returns: Price series or returns
            is_returns: Whether input is returns (True) or prices (False)
            
        Returns:
            Dictionary with drawdown statistics
        """
        if is_returns:
            # Convert returns to cumulative wealth
            cumulative_wealth = np.cumprod(1 + prices_or_returns)
        else:
            cumulative_wealth = prices_or_returns
        
        # Calculate running maximum
        running_max = np.maximum.accumulate(cumulative_wealth)
        
        # Calculate drawdown
        drawdown = (cumulative_wealth - running_max) / running_max
        
        # Find maximum drawdown
        max_dd = np.min(drawdown)
        max_dd_idx = np.argmin(drawdown)
        
        # Find recovery time
        recovery_idx = None
        for i in range(max_dd_idx, len(drawdown)):
            if drawdown[i] >= -0.001:  # Within 0.1% of recovery
                recovery_idx = i
                break
        
        recovery_time = recovery_idx - max_dd_idx if recovery_idx else None
        
        return {
            'max_drawdown': max_dd,
            'max_drawdown_index': max_dd_idx,
            'recovery_time': recovery_time,
            'current_drawdown': drawdown[-1] if len(drawdown) > 0 else 0,
            'average_drawdown': np.mean(drawdown[drawdown < 0]) if np.any(drawdown < 0) else 0
        }
    
    @staticmethod
    def risk_adjusted_returns(returns: Union[np.ndarray, pd.Series],
                             risk_free_rate: float = 0.02) -> Dict[str, float]:
        """
        Calculate risk-adjusted return metrics.
        
        Args:
            returns: Returns series
            risk_free_rate: Risk-free rate (annualized)
            
        Returns:
            Dictionary with risk-adjusted metrics
        """
        if isinstance(returns, pd.Series):
            returns_array = returns.values
        else:
            returns_array = returns
        
        # Annualize returns (assuming daily returns)
        annual_return = np.mean(returns_array) * 252
        annual_vol = np.std(returns_array, ddof=1) * np.sqrt(252)
        
        # Sharpe ratio
        sharpe_ratio = (annual_return - risk_free_rate) / annual_vol if annual_vol > 0 else 0
        
        # Sortino ratio (downside deviation)
        negative_returns = returns_array[returns_array < 0]
        downside_vol = np.std(negative_returns, ddof=1) * np.sqrt(252) if len(negative_returns) > 0 else 0
        sortino_ratio = (annual_return - risk_free_rate) / downside_vol if downside_vol > 0 else 0
        
        # Calmar ratio (return / max drawdown)
        max_dd = RiskMetrics.maximum_drawdown(returns_array, is_returns=True)['max_drawdown']
        calmar_ratio = annual_return / abs(max_dd) if max_dd < 0 else 0
        
        return {
            'annual_return': annual_return,
            'annual_volatility': annual_vol,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'downside_volatility': downside_vol
        }
    
    @staticmethod
    def portfolio_diversification_ratio(returns: pd.DataFrame, 
                                       weights: np.ndarray) -> float:
        """
        Calculate portfolio diversification ratio.
        
        Args:
            returns: DataFrame with asset returns
            weights: Portfolio weights
            
        Returns:
            Diversification ratio
        """
        # Individual asset volatilities
        individual_vols = returns.std() * np.sqrt(252)
        
        # Weighted average of individual volatilities
        weighted_vol = np.sum(weights * individual_vols)
        
        # Portfolio volatility
        portfolio_vol = np.sqrt(weights.T @ returns.cov().values @ weights) * np.sqrt(252)
        
        # Diversification ratio
        diversification_ratio = weighted_vol / portfolio_vol if portfolio_vol > 0 else 0
        
        return diversification_ratio


class RiskDecomposition:
    """
    Risk decomposition and attribution analysis.
    """
    
    @staticmethod
    def marginal_var(returns: pd.DataFrame, 
                    weights: np.ndarray,
                    confidence_level: float = 0.95,
                    delta: float = 0.01) -> np.ndarray:
        """
        Calculate marginal VaR for each asset.
        
        Args:
            returns: DataFrame with asset returns
            weights: Portfolio weights
            confidence_level: Confidence level for VaR
            delta: Small change in weight for numerical derivative
            
        Returns:
            Array of marginal VaR values
        """
        # Calculate baseline portfolio VaR
        portfolio_returns = (returns * weights).sum(axis=1)
        baseline_var = RiskMetrics.value_at_risk(portfolio_returns, confidence_level)
        
        marginal_vars = np.zeros(len(weights))
        
        for i in range(len(weights)):
            # Create perturbed weights
            perturbed_weights = weights.copy()
            perturbed_weights[i] += delta
            
            # Renormalize weights
            perturbed_weights /= perturbed_weights.sum()
            
            # Calculate perturbed portfolio VaR
            perturbed_returns = (returns * perturbed_weights).sum(axis=1)
            perturbed_var = RiskMetrics.value_at_risk(perturbed_returns, confidence_level)
            
            # Calculate marginal VaR
            marginal_vars[i] = (perturbed_var - baseline_var) / delta
        
        return marginal_vars
    
    @staticmethod
    def component_var(returns: pd.DataFrame, 
                     weights: np.ndarray,
                     confidence_level: float = 0.95) -> Dict[str, np.ndarray]:
        """
        Calculate component VaR (risk contribution of each asset).
        
        Args:
            returns: DataFrame with asset returns
            weights: Portfolio weights
            confidence_level: Confidence level for VaR
            
        Returns:
            Dictionary with component VaR analysis
        """
        marginal_vars = RiskDecomposition.marginal_var(returns, weights, confidence_level)
        
        # Component VaR = Weight * Marginal VaR
        component_vars = weights * marginal_vars
        
        # Portfolio VaR for validation
        portfolio_returns = (returns * weights).sum(axis=1)
        portfolio_var = RiskMetrics.value_at_risk(portfolio_returns, confidence_level)
        
        return {
            'marginal_var': marginal_vars,
            'component_var': component_vars,
            'portfolio_var': portfolio_var,
            'var_contributions_pct': component_vars / portfolio_var * 100 if portfolio_var != 0 else np.zeros_like(component_vars)
        }


class RiskReporting:
    """
    Risk reporting and dashboard utilities.
    """
    
    @staticmethod
    def generate_risk_report(returns: Union[np.ndarray, pd.Series, pd.DataFrame],
                           portfolio_value: float = 100000,
                           weights: Optional[np.ndarray] = None) -> Dict[str, any]:
        """
        Generate comprehensive risk report.
        
        Args:
            returns: Returns data
            portfolio_value: Portfolio value for dollar VaR
            weights: Portfolio weights (for DataFrame input)
            
        Returns:
            Comprehensive risk report
        """
        if isinstance(returns, pd.DataFrame):
            if weights is None:
                raise ValueError("Weights required for DataFrame input")
            portfolio_returns = (returns * weights).sum(axis=1)
        else:
            portfolio_returns = returns
        
        # Basic risk metrics
        var_cvar = RiskMetrics.calculate_multiple_vars(
            portfolio_returns,
            confidence_levels=[0.90, 0.95, 0.99],
            methods=["historical", "parametric"]
        )
        
        # Risk-adjusted returns
        risk_adjusted = RiskMetrics.risk_adjusted_returns(portfolio_returns)
        
        # Maximum drawdown
        drawdown_stats = RiskMetrics.maximum_drawdown(portfolio_returns, is_returns=True)
        
        # Convert to dollar terms
        dollar_vars = {}
        for method in var_cvar:
            dollar_vars[method] = {
                key: value * portfolio_value for key, value in var_cvar[method].items()
            }
        
        report = {
            'risk_metrics': {
                'var_cvar_percentage': var_cvar,
                'var_cvar_dollar': dollar_vars,
                'risk_adjusted_returns': risk_adjusted,
                'drawdown_analysis': drawdown_stats
            },
            'portfolio_stats': {
                'total_observations': len(portfolio_returns),
                'portfolio_value': portfolio_value,
                'return_statistics': {
                    'mean_daily': np.mean(portfolio_returns),
                    'std_daily': np.std(portfolio_returns, ddof=1),
                    'skewness': stats.skew(portfolio_returns),
                    'kurtosis': stats.kurtosis(portfolio_returns),
                    'min_return': np.min(portfolio_returns),
                    'max_return': np.max(portfolio_returns)
                }
            }
        }
        
        # Add component analysis if DataFrame input
        if isinstance(returns, pd.DataFrame) and weights is not None:
            component_analysis = RiskDecomposition.component_var(returns, weights)
            report['risk_metrics']['component_analysis'] = component_analysis
            
            # Add diversification ratio
            div_ratio = RiskMetrics.portfolio_diversification_ratio(returns, weights)
            report['portfolio_stats']['diversification_ratio'] = div_ratio
        
        return report