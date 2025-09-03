"""
Monte Carlo simulation engine for portfolio risk modeling.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from scipy.linalg import cholesky
from scipy import stats
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)


class MonteCarloSimulator:
    """
    Monte Carlo simulation engine for portfolio risk analysis.
    """
    
    def __init__(self, 
                 returns: pd.DataFrame,
                 portfolio_weights: np.ndarray,
                 random_seed: Optional[int] = None):
        """
        Initialize Monte Carlo simulator.
        
        Args:
            returns: Historical returns DataFrame
            portfolio_weights: Array of portfolio weights
            random_seed: Random seed for reproducibility
        """
        self.returns = returns
        self.portfolio_weights = portfolio_weights
        self.n_assets = len(portfolio_weights)
        
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # Calculate statistical parameters from historical data
        self.mean_returns = returns.mean().values
        self.cov_matrix = returns.cov().values
        self.correlation_matrix = returns.corr().values
        
        # Calculate Cholesky decomposition for correlated sampling
        try:
            self.chol_matrix = cholesky(self.cov_matrix, lower=True)
        except np.linalg.LinAlgError:
            # Handle non-positive definite matrix
            logger.warning("Covariance matrix is not positive definite. Using regularized version.")
            regularized_cov = self.cov_matrix + np.eye(self.n_assets) * 1e-8
            self.chol_matrix = cholesky(regularized_cov, lower=True)
        
        logger.info(f"Initialized Monte Carlo simulator for {self.n_assets} assets")
    
    def simulate_returns(self, 
                        n_simulations: int = 100000,
                        time_horizon: int = 1,
                        distribution: str = "normal") -> np.ndarray:
        """
        Simulate asset returns using multivariate distribution.
        
        Args:
            n_simulations: Number of Monte Carlo simulations
            time_horizon: Time horizon in days
            distribution: Distribution type ("normal", "t")
            
        Returns:
            Array of simulated returns (n_simulations, n_assets, time_horizon)
        """
        logger.info(f"Running {n_simulations:,} Monte Carlo simulations")
        
        if distribution == "normal":
            return self._simulate_normal_returns(n_simulations, time_horizon)
        elif distribution == "t":
            return self._simulate_t_returns(n_simulations, time_horizon)
        else:
            raise ValueError(f"Unsupported distribution: {distribution}")
    
    def _simulate_normal_returns(self, 
                                n_simulations: int, 
                                time_horizon: int) -> np.ndarray:
        """Simulate returns using multivariate normal distribution."""
        # Generate independent standard normal random variables
        random_normals = np.random.standard_normal(
            (n_simulations, self.n_assets, time_horizon)
        )
        
        # Apply Cholesky decomposition for correlation
        correlated_returns = np.zeros_like(random_normals)
        
        for t in range(time_horizon):
            # Transform independent normals to correlated normals
            correlated_normals = (self.chol_matrix @ random_normals[:, :, t].T).T
            
            # Scale by mean and standard deviation
            correlated_returns[:, :, t] = (
                self.mean_returns[np.newaxis, :] + correlated_normals
            )
        
        return correlated_returns
    
    def _simulate_t_returns(self, 
                           n_simulations: int, 
                           time_horizon: int,
                           df: float = 5.0) -> np.ndarray:
        """Simulate returns using multivariate t-distribution."""
        # Generate t-distributed random variables
        chi2_samples = np.random.chisquare(df, (n_simulations, time_horizon))
        normal_samples = np.random.standard_normal(
            (n_simulations, self.n_assets, time_horizon)
        )
        
        # Scale to get t-distribution
        t_samples = normal_samples * np.sqrt(df / chi2_samples[:, np.newaxis, :])
        
        # Apply correlation structure
        correlated_returns = np.zeros_like(t_samples)
        
        for t in range(time_horizon):
            correlated_normals = (self.chol_matrix @ t_samples[:, :, t].T).T
            correlated_returns[:, :, t] = (
                self.mean_returns[np.newaxis, :] + correlated_normals
            )
        
        return correlated_returns
    
    def simulate_portfolio_returns(self, 
                                  n_simulations: int = 100000,
                                  time_horizon: int = 1,
                                  distribution: str = "normal") -> np.ndarray:
        """
        Simulate portfolio returns.
        
        Args:
            n_simulations: Number of Monte Carlo simulations
            time_horizon: Time horizon in days
            distribution: Distribution type
            
        Returns:
            Array of simulated portfolio returns
        """
        asset_returns = self.simulate_returns(n_simulations, time_horizon, distribution)
        
        # Calculate portfolio returns for each simulation and time step
        portfolio_returns = np.sum(
            asset_returns * self.portfolio_weights[np.newaxis, :, np.newaxis],
            axis=1
        )
        
        # If time horizon > 1, calculate cumulative returns
        if time_horizon > 1:
            portfolio_returns = np.prod(1 + portfolio_returns, axis=1) - 1
        else:
            portfolio_returns = portfolio_returns.squeeze()
        
        return portfolio_returns
    
    def simulate_portfolio_pnl(self, 
                              portfolio_value: float,
                              n_simulations: int = 100000,
                              time_horizon: int = 1,
                              distribution: str = "normal") -> np.ndarray:
        """
        Simulate portfolio P&L in dollar terms.
        
        Args:
            portfolio_value: Initial portfolio value
            n_simulations: Number of simulations
            time_horizon: Time horizon in days
            distribution: Distribution type
            
        Returns:
            Array of simulated P&L values
        """
        portfolio_returns = self.simulate_portfolio_returns(
            n_simulations, time_horizon, distribution
        )
        
        pnl = portfolio_value * portfolio_returns
        return pnl
    
    def run_stress_test(self, 
                       stress_scenarios: Dict[str, Dict[str, float]],
                       portfolio_value: float,
                       n_simulations: int = 10000) -> Dict[str, np.ndarray]:
        """
        Run stress tests with modified market conditions.
        
        Args:
            stress_scenarios: Dictionary of stress scenarios
            portfolio_value: Initial portfolio value
            n_simulations: Number of simulations per scenario
            
        Returns:
            Dictionary of stress test results
        """
        results = {}
        
        for scenario_name, scenario_params in stress_scenarios.items():
            logger.info(f"Running stress test: {scenario_name}")
            
            # Create modified parameters for stress test
            stress_mean = self.mean_returns.copy()
            stress_cov = self.cov_matrix.copy()
            
            # Apply stress modifications
            if 'vol_multiplier' in scenario_params:
                stress_cov *= scenario_params['vol_multiplier']
            
            if 'return_shock' in scenario_params:
                stress_mean += scenario_params['return_shock']
            
            if 'correlation_multiplier' in scenario_params:
                # Modify correlation while preserving variances
                corr = self.correlation_matrix * scenario_params['correlation_multiplier']
                np.fill_diagonal(corr, 1.0)  # Keep diagonal as 1
                
                # Convert back to covariance matrix
                vol = np.sqrt(np.diag(stress_cov))
                stress_cov = np.outer(vol, vol) * corr
            
            # Create temporary simulator with stress parameters
            temp_returns = pd.DataFrame(
                np.random.multivariate_normal(stress_mean, stress_cov, size=100),
                columns=self.returns.columns
            )
            
            stress_simulator = MonteCarloSimulator(
                temp_returns, self.portfolio_weights
            )
            
            # Override with stress parameters
            stress_simulator.mean_returns = stress_mean
            stress_simulator.cov_matrix = stress_cov
            stress_simulator.chol_matrix = cholesky(stress_cov, lower=True)
            
            # Run simulation
            stress_pnl = stress_simulator.simulate_portfolio_pnl(
                portfolio_value, n_simulations
            )
            
            results[scenario_name] = stress_pnl
        
        return results
    
    def calculate_confidence_intervals(self, 
                                     simulated_values: np.ndarray,
                                     confidence_levels: List[float] = [0.95, 0.99]) -> Dict[str, float]:
        """
        Calculate confidence intervals for simulated values.
        
        Args:
            simulated_values: Array of simulated values
            confidence_levels: List of confidence levels
            
        Returns:
            Dictionary with confidence interval bounds
        """
        results = {}
        
        for conf_level in confidence_levels:
            alpha = 1 - conf_level
            lower_percentile = (alpha / 2) * 100
            upper_percentile = (1 - alpha / 2) * 100
            
            results[f'ci_{conf_level}_lower'] = np.percentile(
                simulated_values, lower_percentile
            )
            results[f'ci_{conf_level}_upper'] = np.percentile(
                simulated_values, upper_percentile
            )
        
        return results
    
    def get_simulation_summary(self, 
                              simulated_values: np.ndarray) -> Dict[str, float]:
        """
        Get summary statistics for simulation results.
        
        Args:
            simulated_values: Array of simulated values
            
        Returns:
            Dictionary with summary statistics
        """
        return {
            'mean': np.mean(simulated_values),
            'std': np.std(simulated_values),
            'min': np.min(simulated_values),
            'max': np.max(simulated_values),
            'median': np.median(simulated_values),
            'skewness': stats.skew(simulated_values),
            'kurtosis': stats.kurtosis(simulated_values),
            'var_95': np.percentile(simulated_values, 5),
            'var_99': np.percentile(simulated_values, 1),
            'cvar_95': np.mean(simulated_values[simulated_values <= np.percentile(simulated_values, 5)]),
            'cvar_99': np.mean(simulated_values[simulated_values <= np.percentile(simulated_values, 1)])
        }