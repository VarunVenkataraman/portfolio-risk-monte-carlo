"""
Data loading module for financial data acquisition and preprocessing.
"""
import pandas as pd
import yfinance as yf
import numpy as np
from typing import List, Dict, Tuple, Optional
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataLoader:
    """
    Handles financial data loading and preprocessing for portfolio risk analysis.
    """
    
    def __init__(self, cache_dir: str = "data/raw"):
        """
        Initialize DataLoader with cache directory.
        
        Args:
            cache_dir: Directory to cache downloaded data
        """
        self.cache_dir = cache_dir
        
    def fetch_etf_data(self, 
                      tickers: List[str], 
                      start_date: str, 
                      end_date: str,
                      use_cache: bool = True) -> pd.DataFrame:
        """
        Fetch ETF price data from Yahoo Finance.
        
        Args:
            tickers: List of ETF ticker symbols
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            use_cache: Whether to use cached data if available
            
        Returns:
            DataFrame with adjusted closing prices
        """
        cache_file = f"{self.cache_dir}/etf_prices_{start_date}_{end_date}.csv"
        
        if use_cache:
            try:
                logger.info(f"Loading cached data from {cache_file}")
                return pd.read_csv(cache_file, index_col=0, parse_dates=True)
            except FileNotFoundError:
                pass
        
        logger.info(f"Fetching data for {tickers} from {start_date} to {end_date}")
        
        try:
            data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
            
            if len(tickers) == 1:
                data = data.to_frame(tickers[0])
            
            # Remove any rows with all NaN values
            data = data.dropna(how='all')
            
            # Forward fill any remaining NaN values
            data = data.fillna(method='ffill')
            
            # Save to cache
            import os
            os.makedirs(self.cache_dir, exist_ok=True)
            data.to_csv(cache_file)
            logger.info(f"Data cached to {cache_file}")
            
            return data
            
        except Exception as e:
            logger.error(f"Error fetching data: {e}")
            raise
    
    def calculate_returns(self, prices: pd.DataFrame, return_type: str = "simple") -> pd.DataFrame:
        """
        Calculate returns from price data.
        
        Args:
            prices: DataFrame with price data
            return_type: Type of returns ('simple' or 'log')
            
        Returns:
            DataFrame with returns
        """
        if return_type == "simple":
            returns = prices.pct_change()
        elif return_type == "log":
            returns = np.log(prices / prices.shift(1))
        else:
            raise ValueError("return_type must be 'simple' or 'log'")
        
        return returns.dropna()
    
    def get_portfolio_data(self, 
                          tickers: List[str] = None,
                          years_back: int = 5,
                          return_type: str = "simple") -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Get portfolio data with default ETF universe.
        
        Args:
            tickers: List of ticker symbols (uses default if None)
            years_back: Number of years of historical data
            return_type: Type of returns to calculate
            
        Returns:
            Tuple of (prices, returns) DataFrames
        """
        if tickers is None:
            tickers = ['SPY', 'EFA', 'EEM', 'TLT', 'LQD', 'GLD', 'VNQ']
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=years_back * 365)
        
        prices = self.fetch_etf_data(
            tickers=tickers,
            start_date=start_date.strftime('%Y-%m-%d'),
            end_date=end_date.strftime('%Y-%m-%d')
        )
        
        returns = self.calculate_returns(prices, return_type)
        
        logger.info(f"Loaded data for {len(tickers)} ETFs")
        logger.info(f"Price data shape: {prices.shape}")
        logger.info(f"Returns data shape: {returns.shape}")
        logger.info(f"Date range: {prices.index.min()} to {prices.index.max()}")
        
        return prices, returns
    
    def validate_data(self, data: pd.DataFrame) -> Dict[str, any]:
        """
        Validate data quality and return summary statistics.
        
        Args:
            data: DataFrame to validate
            
        Returns:
            Dictionary with validation results
        """
        validation_results = {
            'shape': data.shape,
            'date_range': (data.index.min(), data.index.max()),
            'missing_values': data.isnull().sum().to_dict(),
            'columns': list(data.columns),
            'data_types': data.dtypes.to_dict()
        }
        
        # Check for suspicious returns (> 50% daily move)
        if any('return' in col.lower() for col in data.columns):
            extreme_moves = (abs(data) > 0.5).sum()
            validation_results['extreme_moves'] = extreme_moves.to_dict()
        
        return validation_results