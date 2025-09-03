"""
Configuration and utility functions.
"""
import os
import yaml
from typing import Dict, List, Any
from datetime import datetime, timedelta


class Config:
    """Configuration management."""
    
    # Default ETF universe
    DEFAULT_ETFS = ['SPY', 'EFA', 'EEM', 'TLT', 'LQD', 'GLD', 'VNQ']
    
    # Default portfolio settings
    DEFAULT_PORTFOLIO_VALUE = 100000.0
    DEFAULT_CONFIDENCE_LEVELS = [0.90, 0.95, 0.99]
    
    # Simulation settings
    DEFAULT_SIMULATIONS = 100000
    DEFAULT_TIME_HORIZON = 1  # days
    
    # Historical data settings
    DEFAULT_YEARS_BACK = 5
    DEFAULT_RETURN_TYPE = "simple"
    
    # Risk-free rate (annual)
    DEFAULT_RISK_FREE_RATE = 0.02
    
    # Crisis periods for historical stress testing
    CRISIS_PERIODS = {
        'covid_crash': ('2020-02-20', '2020-04-01'),
        'financial_crisis': ('2008-09-01', '2009-03-31'),
        'dot_com_crash': ('2000-03-01', '2002-10-01'),
        'brexit_vote': ('2016-06-23', '2016-07-01'),
        'flash_crash': ('2010-05-06', '2010-05-07')
    }
    
    # Stress test scenarios
    STRESS_SCENARIOS = {
        'high_volatility': {
            'volatility_multipliers': {etf: 2.0 for etf in DEFAULT_ETFS},
            'description': 'Double volatility across all assets'
        },
        'correlation_breakdown': {
            'correlation_multiplier': 0.1,
            'description': 'Correlation approaches zero'
        },
        'market_crash': {
            'return_shocks': {etf: -0.05 for etf in DEFAULT_ETFS},
            'volatility_multipliers': {etf: 1.5 for etf in DEFAULT_ETFS},
            'description': '5% return shock with 50% volatility increase'
        },
        'flight_to_quality': {
            'return_shocks': {
                'SPY': -0.03, 'EFA': -0.04, 'EEM': -0.06,
                'TLT': 0.02, 'LQD': 0.01, 'GLD': 0.01, 'VNQ': -0.03
            },
            'description': 'Flight to bonds and gold, equity selloff'
        }
    }


def create_project_structure(base_dir: str = ".") -> None:
    """Create project directory structure."""
    directories = [
        "src/data",
        "src/portfolio", 
        "src/simulation",
        "src/risk",
        "src/backtesting",
        "src/utils",
        "notebooks",
        "tests",
        "data/raw",
        "data/processed",
        "docs",
        "results"
    ]
    
    for directory in directories:
        full_path = os.path.join(base_dir, directory)
        os.makedirs(full_path, exist_ok=True)
        
        # Create __init__.py for Python packages
        if directory.startswith("src/"):
            init_file = os.path.join(full_path, "__init__.py")
            if not os.path.exists(init_file):
                with open(init_file, 'w') as f:
                    f.write("")


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        return {}


def save_config(config: Dict[str, Any], config_path: str) -> None:
    """Save configuration to YAML file."""
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


def get_date_range(years_back: int = 5) -> tuple[str, str]:
    """Get date range for data fetching."""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=years_back * 365)
    return start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')


def format_currency(amount: float) -> str:
    """Format amount as currency."""
    return f"${amount:,.2f}"


def format_percentage(value: float, decimals: int = 2) -> str:
    """Format value as percentage."""
    return f"{value * 100:.{decimals}f}%"