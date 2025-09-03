"""
Backtesting and model validation module for VaR models.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from scipy import stats
import logging
from ..risk.metrics import RiskMetrics

logger = logging.getLogger(__name__)


class VaRBacktest:
    """
    Value at Risk backtesting and validation.
    """
    
    def __init__(self, 
                 returns: pd.Series,
                 var_forecasts: pd.Series,
                 confidence_level: float = 0.95):
        """
        Initialize VaR backtest.
        
        Args:
            returns: Actual portfolio returns
            var_forecasts: VaR forecasts (negative values for losses)
            confidence_level: Confidence level used for VaR
        """
        self.returns = returns
        self.var_forecasts = var_forecasts
        self.confidence_level = confidence_level
        self.expected_violations = 1 - confidence_level
        
        # Align data
        common_index = self.returns.index.intersection(self.var_forecasts.index)
        self.returns = self.returns.loc[common_index]
        self.var_forecasts = self.var_forecasts.loc[common_index]
        
        # Calculate violations (when actual return < VaR forecast)
        self.violations = (self.returns <= self.var_forecasts).astype(int)
        self.n_observations = len(self.returns)
        self.n_violations = self.violations.sum()
        self.violation_rate = self.n_violations / self.n_observations
        
        logger.info(f"Backtest initialized: {self.n_observations} observations, {self.n_violations} violations")
    
    def kupiec_pof_test(self) -> Dict[str, float]:
        """
        Kupiec Proportion of Failures (POF) test.
        
        Tests if the violation rate is significantly different from expected.
        
        Returns:
            Dictionary with test results
        """
        n = self.n_observations
        x = self.n_violations
        p = self.expected_violations
        
        if x == 0:
            lr_stat = -2 * n * np.log(1 - p)
        elif x == n:
            lr_stat = -2 * n * np.log(p)
        else:
            # Likelihood ratio test statistic
            lr_stat = -2 * (n * np.log(1 - p) + x * np.log(p / (x/n)) + (n - x) * np.log((1-p)/(1-x/n)))
        
        # Critical value for 95% confidence (chi-square with 1 df)
        critical_value = 3.841
        p_value = 1 - stats.chi2.cdf(lr_stat, df=1)
        
        return {
            'test_statistic': lr_stat,
            'critical_value': critical_value,
            'p_value': p_value,
            'reject_null': lr_stat > critical_value,
            'expected_violations': int(n * p),
            'actual_violations': x,
            'violation_rate': self.violation_rate,
            'expected_rate': p
        }
    
    def christoffersen_independence_test(self) -> Dict[str, float]:
        """
        Christoffersen independence test for clustering of violations.
        
        Tests if violations are independently distributed over time.
        
        Returns:
            Dictionary with test results
        """
        violations = self.violations.values
        n = len(violations)
        
        # Count transitions
        n00 = n01 = n10 = n11 = 0
        
        for i in range(n - 1):
            if violations[i] == 0 and violations[i + 1] == 0:
                n00 += 1
            elif violations[i] == 0 and violations[i + 1] == 1:
                n01 += 1
            elif violations[i] == 1 and violations[i + 1] == 0:
                n10 += 1
            elif violations[i] == 1 and violations[i + 1] == 1:
                n11 += 1
        
        # Calculate test statistic
        if n01 + n11 == 0 or n00 + n10 == 0:
            # No violations or all violations
            lr_ind = 0
        else:
            pi01 = n01 / (n00 + n01) if (n00 + n01) > 0 else 0
            pi11 = n11 / (n10 + n11) if (n10 + n11) > 0 else 0
            pi = (n01 + n11) / (n - 1) if (n - 1) > 0 else 0
            
            if pi01 == 0 or pi11 == 0 or pi == 0:
                lr_ind = 0
            else:
                lr_ind = -2 * (
                    n01 * np.log(pi / pi01) + 
                    n11 * np.log(pi / pi11) + 
                    n00 * np.log((1 - pi) / (1 - pi01)) + 
                    n10 * np.log((1 - pi) / (1 - pi11))
                )
        
        critical_value = 3.841  # 95% confidence, 1 df
        p_value = 1 - stats.chi2.cdf(lr_ind, df=1)
        
        return {
            'test_statistic': lr_ind,
            'critical_value': critical_value,
            'p_value': p_value,
            'reject_null': lr_ind > critical_value,
            'transition_matrix': {
                '00': n00, '01': n01,
                '10': n10, '11': n11
            }
        }
    
    def christoffersen_joint_test(self) -> Dict[str, float]:
        """
        Christoffersen joint test (coverage + independence).
        
        Returns:
            Dictionary with joint test results
        """
        pof_result = self.kupiec_pof_test()
        ind_result = self.christoffersen_independence_test()
        
        joint_stat = pof_result['test_statistic'] + ind_result['test_statistic']
        critical_value = 5.991  # 95% confidence, 2 df
        p_value = 1 - stats.chi2.cdf(joint_stat, df=2)
        
        return {
            'joint_statistic': joint_stat,
            'pof_statistic': pof_result['test_statistic'],
            'independence_statistic': ind_result['test_statistic'],
            'critical_value': critical_value,
            'p_value': p_value,
            'reject_null': joint_stat > critical_value
        }
    
    def calculate_violation_metrics(self) -> Dict[str, float]:
        """
        Calculate additional violation-based metrics.
        
        Returns:
            Dictionary with violation metrics
        """
        violations = self.violations.values
        
        # Average time between violations
        violation_indices = np.where(violations == 1)[0]
        if len(violation_indices) > 1:
            time_between_violations = np.diff(violation_indices)
            avg_time_between = np.mean(time_between_violations)
            max_time_between = np.max(time_between_violations)
        else:
            avg_time_between = max_time_between = np.nan
        
        # Longest violation streak
        streaks = []
        current_streak = 0
        for v in violations:
            if v == 1:
                current_streak += 1
            else:
                if current_streak > 0:
                    streaks.append(current_streak)
                current_streak = 0
        if current_streak > 0:
            streaks.append(current_streak)
        
        max_streak = max(streaks) if streaks else 0
        
        return {
            'avg_time_between_violations': avg_time_between,
            'max_time_between_violations': max_time_between,
            'max_violation_streak': max_streak,
            'total_violation_streaks': len(streaks)
        }
    
    def rolling_backtest(self, window_size: int = 252) -> pd.DataFrame:
        """
        Perform rolling backtest analysis.
        
        Args:
            window_size: Rolling window size in days
            
        Returns:
            DataFrame with rolling backtest results
        """
        results = []
        
        for i in range(window_size, len(self.returns)):
            window_returns = self.returns.iloc[i-window_size:i]
            window_var = self.var_forecasts.iloc[i-window_size:i]
            
            # Create temporary backtest for this window
            temp_backtest = VaRBacktest(window_returns, window_var, self.confidence_level)
            
            pof_result = temp_backtest.kupiec_pof_test()
            
            results.append({
                'date': self.returns.index[i],
                'violation_rate': temp_backtest.violation_rate,
                'violations': temp_backtest.n_violations,
                'pof_statistic': pof_result['test_statistic'],
                'pof_p_value': pof_result['p_value'],
                'model_adequate': not pof_result['reject_null']
            })
        
        return pd.DataFrame(results).set_index('date')


class StressTesting:
    """
    Stress testing framework for portfolio risk models.
    """
    
    def __init__(self, returns: pd.DataFrame, portfolio_weights: np.ndarray):
        """
        Initialize stress testing framework.
        
        Args:
            returns: Historical returns DataFrame
            portfolio_weights: Portfolio weights
        """
        self.returns = returns
        self.portfolio_weights = portfolio_weights
        self.base_correlation = returns.corr()
        self.base_volatility = returns.std()
        
    def scenario_analysis(self, scenarios: Dict[str, Dict[str, any]]) -> Dict[str, Dict[str, float]]:
        """
        Run scenario analysis with predefined scenarios.
        
        Args:
            scenarios: Dictionary of scenarios with parameters
            
        Returns:
            Dictionary with scenario results
        """
        results = {}
        
        # Calculate baseline metrics
        baseline_returns = (self.returns * self.portfolio_weights).sum(axis=1)
        baseline_var_95 = RiskMetrics.value_at_risk(baseline_returns, 0.95)
        baseline_var_99 = RiskMetrics.value_at_risk(baseline_returns, 0.99)
        
        for scenario_name, params in scenarios.items():
            logger.info(f"Running scenario: {scenario_name}")
            
            # Apply scenario modifications
            stressed_returns = self.returns.copy()
            
            # Apply return shocks
            if 'return_shocks' in params:
                for asset, shock in params['return_shocks'].items():
                    if asset in stressed_returns.columns:
                        stressed_returns[asset] += shock
            
            # Apply volatility scaling
            if 'volatility_multipliers' in params:
                for asset, multiplier in params['volatility_multipliers'].items():
                    if asset in stressed_returns.columns:
                        mean_return = stressed_returns[asset].mean()
                        stressed_returns[asset] = mean_return + (stressed_returns[asset] - mean_return) * multiplier
            
            # Calculate scenario portfolio returns
            scenario_portfolio_returns = (stressed_returns * self.portfolio_weights).sum(axis=1)
            
            # Calculate risk metrics
            var_95 = RiskMetrics.value_at_risk(scenario_portfolio_returns, 0.95)
            var_99 = RiskMetrics.value_at_risk(scenario_portfolio_returns, 0.99)
            cvar_95 = RiskMetrics.conditional_value_at_risk(scenario_portfolio_returns, 0.95)
            cvar_99 = RiskMetrics.conditional_value_at_risk(scenario_portfolio_returns, 0.99)
            
            results[scenario_name] = {
                'var_95': var_95,
                'var_99': var_99,
                'cvar_95': cvar_95,
                'cvar_99': cvar_99,
                'var_95_change': var_95 - baseline_var_95,
                'var_99_change': var_99 - baseline_var_99,
                'annual_return': scenario_portfolio_returns.mean() * 252,
                'annual_volatility': scenario_portfolio_returns.std() * np.sqrt(252)
            }
        
        return results
    
    def sensitivity_analysis(self, 
                           factor_ranges: Dict[str, Tuple[float, float]],
                           n_steps: int = 10) -> pd.DataFrame:
        """
        Perform sensitivity analysis across factor ranges.
        
        Args:
            factor_ranges: Dictionary with factor names and (min, max) ranges
            n_steps: Number of steps in each range
            
        Returns:
            DataFrame with sensitivity results
        """
        results = []
        baseline_returns = (self.returns * self.portfolio_weights).sum(axis=1)
        baseline_var = RiskMetrics.value_at_risk(baseline_returns, 0.95)
        
        for factor_name, (min_val, max_val) in factor_ranges.items():
            factor_values = np.linspace(min_val, max_val, n_steps)
            
            for factor_value in factor_values:
                # Apply factor stress
                if factor_name == 'correlation_multiplier':
                    # Modify correlation matrix
                    stressed_corr = self.base_correlation * factor_value
                    np.fill_diagonal(stressed_corr.values, 1.0)
                    
                    # Generate stressed returns (simplified approach)
                    stressed_returns = self.returns.copy()
                    
                elif factor_name == 'volatility_multiplier':
                    # Scale volatility
                    stressed_returns = self.returns.copy()
                    for col in stressed_returns.columns:
                        mean_ret = stressed_returns[col].mean()
                        stressed_returns[col] = mean_ret + (stressed_returns[col] - mean_ret) * factor_value
                
                elif factor_name == 'market_shock':
                    # Apply uniform market shock
                    stressed_returns = self.returns + factor_value
                
                else:
                    continue
                
                # Calculate stressed VaR
                stressed_portfolio_returns = (stressed_returns * self.portfolio_weights).sum(axis=1)
                stressed_var = RiskMetrics.value_at_risk(stressed_portfolio_returns, 0.95)
                
                results.append({
                    'factor': factor_name,
                    'factor_value': factor_value,
                    'var_95': stressed_var,
                    'var_change': stressed_var - baseline_var,
                    'var_change_pct': (stressed_var - baseline_var) / abs(baseline_var) * 100
                })
        
        return pd.DataFrame(results)
    
    def historical_scenarios(self, 
                           crisis_periods: Dict[str, Tuple[str, str]]) -> Dict[str, Dict[str, float]]:
        """
        Apply historical crisis scenarios to current portfolio.
        
        Args:
            crisis_periods: Dictionary with crisis names and date ranges
            
        Returns:
            Dictionary with historical scenario results
        """
        results = {}
        
        for crisis_name, (start_date, end_date) in crisis_periods.items():
            try:
                # Extract crisis period returns
                crisis_returns = self.returns.loc[start_date:end_date]
                
                if len(crisis_returns) == 0:
                    logger.warning(f"No data found for {crisis_name} period")
                    continue
                
                # Apply crisis returns to current portfolio
                crisis_portfolio_returns = (crisis_returns * self.portfolio_weights).sum(axis=1)
                
                # Calculate metrics
                var_95 = RiskMetrics.value_at_risk(crisis_portfolio_returns, 0.95)
                var_99 = RiskMetrics.value_at_risk(crisis_portfolio_returns, 0.99)
                max_dd = RiskMetrics.maximum_drawdown(crisis_portfolio_returns, is_returns=True)
                
                results[crisis_name] = {
                    'period': f"{start_date} to {end_date}",
                    'days': len(crisis_portfolio_returns),
                    'var_95': var_95,
                    'var_99': var_99,
                    'worst_day': crisis_portfolio_returns.min(),
                    'best_day': crisis_portfolio_returns.max(),
                    'max_drawdown': max_dd['max_drawdown'],
                    'total_return': (1 + crisis_portfolio_returns).prod() - 1,
                    'volatility': crisis_portfolio_returns.std() * np.sqrt(252)
                }
                
            except Exception as e:
                logger.error(f"Error processing {crisis_name}: {e}")
                continue
        
        return results