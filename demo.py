"""
Demo version of Portfolio Risk Monte Carlo Simulation using simulated data.

This script demonstrates the complete workflow when Yahoo Finance API is unavailable.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta

# Import project modules
from src.portfolio.construction import Portfolio, PortfolioAnalyzer
from src.simulation.monte_carlo import MonteCarloSimulator
from src.risk.metrics import RiskReporting
from src.backtesting.validation import StressTesting
from src.utils.config import Config, format_currency, format_percentage

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_simulated_data(tickers, n_days=1260, start_date='2020-01-01'):
    """Generate simulated market data for demonstration."""
    print("🎲 Generating simulated market data for demonstration...")
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Create realistic market parameters
    annual_returns = {
        'SPY': 0.10,   # S&P 500
        'EFA': 0.06,   # Developed International
        'EEM': 0.05,   # Emerging Markets
        'TLT': 0.03,   # Long-term Treasuries
        'LQD': 0.04,   # Investment Grade Bonds
        'GLD': 0.05,   # Gold
        'VNQ': 0.08    # REITs
    }
    
    annual_volatilities = {
        'SPY': 0.16,
        'EFA': 0.18,
        'EEM': 0.24,
        'TLT': 0.12,
        'LQD': 0.08,
        'GLD': 0.20,
        'VNQ': 0.22
    }
    
    # Correlation matrix (realistic relationships)
    correlation_matrix = np.array([
        [1.00, 0.80, 0.70, -0.20, -0.10, 0.10, 0.60],  # SPY
        [0.80, 1.00, 0.75, -0.15, -0.05, 0.15, 0.55],  # EFA
        [0.70, 0.75, 1.00, -0.10, 0.00, 0.20, 0.50],   # EEM
        [-0.20, -0.15, -0.10, 1.00, 0.70, 0.30, -0.15], # TLT
        [-0.10, -0.05, 0.00, 0.70, 1.00, 0.25, -0.05], # LQD
        [0.10, 0.15, 0.20, 0.30, 0.25, 1.00, 0.20],    # GLD
        [0.60, 0.55, 0.50, -0.15, -0.05, 0.20, 1.00]   # VNQ
    ])
    
    # Generate dates
    dates = pd.date_range(start=start_date, periods=n_days, freq='B')  # Business days
    
    # Convert annual parameters to daily
    daily_returns = np.array([annual_returns[ticker] / 252 for ticker in tickers])
    daily_vols = np.array([annual_volatilities[ticker] / np.sqrt(252) for ticker in tickers])
    
    # Generate correlated returns
    chol = np.linalg.cholesky(correlation_matrix)
    random_normals = np.random.standard_normal((n_days, len(tickers)))
    correlated_returns = random_normals @ chol.T
    
    # Scale by volatility and add drift
    returns_data = {}
    prices_data = {}
    
    for i, ticker in enumerate(tickers):
        # Generate returns
        returns = daily_returns[i] + daily_vols[i] * correlated_returns[:, i]
        returns_data[ticker] = returns
        
        # Generate prices (starting at $100)
        prices = [100.0]
        for ret in returns:
            prices.append(prices[-1] * (1 + ret))
        
        prices_data[ticker] = prices[1:]  # Remove initial price
    
    # Create DataFrames
    returns_df = pd.DataFrame(returns_data, index=dates)
    prices_df = pd.DataFrame(prices_data, index=dates)
    
    return prices_df, returns_df


def main():
    """Main demo function."""
    print("=" * 80)
    print("PORTFOLIO RISK MONTE CARLO SIMULATION - DEMO VERSION")
    print("=" * 80)
    print(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Portfolio Value: {format_currency(Config.DEFAULT_PORTFOLIO_VALUE)}")
    print(f"ETF Universe: {', '.join(Config.DEFAULT_ETFS)}")
    print("📌 Using simulated data (Yahoo Finance API unavailable)")
    print("=" * 80)
    
    try:
        # Step 1: Generate Simulated Data
        print("\n📊 STEP 1: GENERATING SIMULATED MARKET DATA")
        print("-" * 40)
        
        prices, returns = generate_simulated_data(Config.DEFAULT_ETFS, n_days=1260)
        
        print(f"✓ Generated {len(returns)} days of simulated data")
        print(f"✓ Date range: {returns.index.min().date()} to {returns.index.max().date()}")
        print(f"✓ Assets: {len(returns.columns)}")
        
        # Step 2: Portfolio Construction
        print("\n🏗️ STEP 2: PORTFOLIO CONSTRUCTION")
        print("-" * 40)
        
        portfolio = Portfolio(
            tickers=Config.DEFAULT_ETFS,
            weights=None,  # Equal weighting
            notional_value=Config.DEFAULT_PORTFOLIO_VALUE
        )
        
        print(f"✓ Portfolio created with equal weights ({1/len(Config.DEFAULT_ETFS):.3f} each)")
        
        # Portfolio analysis
        analyzer = PortfolioAnalyzer(portfolio)
        performance_report = analyzer.generate_performance_report(returns)
        
        print(f"✓ Portfolio Sharpe Ratio: {performance_report['portfolio_stats']['sharpe_ratio']:.3f}")
        print(f"✓ Portfolio Annual Return: {format_percentage(performance_report['portfolio_stats']['annual_return'])}")
        print(f"✓ Portfolio Annual Volatility: {format_percentage(performance_report['portfolio_stats']['annual_volatility'])}")
        
        # Step 3: Monte Carlo Simulation
        print("\n🎲 STEP 3: MONTE CARLO SIMULATION")
        print("-" * 40)
        
        simulator = MonteCarloSimulator(
            returns=returns[portfolio.tickers],
            portfolio_weights=portfolio.get_weights_array(),
            random_seed=42  # For reproducibility
        )
        
        print(f"✓ Simulator initialized with {len(portfolio.tickers)} assets")
        
        # Run Monte Carlo simulations
        n_simulations = 50000  # Reduced for demo speed
        print(f"🔄 Running {n_simulations:,} Monte Carlo simulations...")
        
        pnl_simulations = simulator.simulate_portfolio_pnl(
            portfolio_value=Config.DEFAULT_PORTFOLIO_VALUE,
            n_simulations=n_simulations,
            time_horizon=Config.DEFAULT_TIME_HORIZON
        )
        
        print(f"✅ Simulations complete")
        
        # Step 4: Risk Metrics Calculation
        print("\n⚠️ STEP 4: RISK METRICS CALCULATION")
        print("-" * 40)
        
        # Generate comprehensive risk report
        portfolio_returns = portfolio.calculate_portfolio_returns(returns)
        
        risk_report = RiskReporting.generate_risk_report(
            returns=portfolio_returns,
            portfolio_value=Config.DEFAULT_PORTFOLIO_VALUE
        )
        
        # Monte Carlo VaR/CVaR
        mc_summary = simulator.get_simulation_summary(pnl_simulations)
        
        print("📈 HISTORICAL RISK METRICS:")
        for conf_level in [90, 95, 99]:
            historical_var = risk_report['risk_metrics']['var_cvar_dollar']['historical'][f'var_{conf_level}']
            historical_cvar = risk_report['risk_metrics']['var_cvar_dollar']['historical'][f'cvar_{conf_level}']
            print(f"  • {conf_level}% VaR: {format_currency(historical_var)}")
            print(f"  • {conf_level}% CVaR: {format_currency(historical_cvar)}")
        
        print("\n🎯 MONTE CARLO RISK METRICS:")
        print(f"  • 95% VaR: {format_currency(mc_summary['var_95'])}")
        print(f"  • 95% CVaR: {format_currency(mc_summary['cvar_95'])}")
        print(f"  • 99% VaR: {format_currency(mc_summary['var_99'])}")
        print(f"  • 99% CVaR: {format_currency(mc_summary['cvar_99'])}")
        
        print("\n📊 PORTFOLIO STATISTICS:")
        print(f"  • Maximum Drawdown: {format_percentage(risk_report['risk_metrics']['drawdown_analysis']['max_drawdown'])}")
        print(f"  • Sharpe Ratio: {risk_report['risk_metrics']['risk_adjusted_returns']['sharpe_ratio']:.3f}")
        print(f"  • Sortino Ratio: {risk_report['risk_metrics']['risk_adjusted_returns']['sortino_ratio']:.3f}")
        
        # Step 5: Stress Testing
        print("\n🔥 STEP 5: STRESS TESTING")
        print("-" * 40)
        
        stress_tester = StressTesting(returns[portfolio.tickers], portfolio.get_weights_array())
        
        # Run predefined stress scenarios
        stress_results = stress_tester.scenario_analysis(Config.STRESS_SCENARIOS)
        
        print("STRESS TEST RESULTS:")
        for scenario_name, results in stress_results.items():
            print(f"\n📍 {scenario_name.replace('_', ' ').title()}:")
            print(f"   • 95% VaR: {format_currency(results['var_95'] * Config.DEFAULT_PORTFOLIO_VALUE)}")
            print(f"   • 99% VaR: {format_currency(results['var_99'] * Config.DEFAULT_PORTFOLIO_VALUE)}")
            print(f"   • Annual Volatility: {format_percentage(results['annual_volatility'])}")
        
        # Step 6: Summary and Recommendations
        print("\n" + "=" * 80)
        print("📋 EXECUTIVE SUMMARY")
        print("=" * 80)
        
        worst_case_var_99 = mc_summary['var_99']
        worst_case_cvar_99 = mc_summary['cvar_99']
        
        print(f"Portfolio Value: {format_currency(Config.DEFAULT_PORTFOLIO_VALUE)}")
        print(f"Time Horizon: {Config.DEFAULT_TIME_HORIZON} day(s)")
        print(f"Monte Carlo Simulations: {n_simulations:,}")
        print(f"")
        print(f"💰 Maximum Expected Loss (99% VaR): {format_currency(worst_case_var_99)}")
        print(f"💸 Average Loss Beyond VaR (99% CVaR): {format_currency(worst_case_cvar_99)}")
        print(f"📊 Portfolio Sharpe Ratio: {risk_report['risk_metrics']['risk_adjusted_returns']['sharpe_ratio']:.3f}")
        print(f"📉 Maximum Historical Drawdown: {format_percentage(risk_report['risk_metrics']['drawdown_analysis']['max_drawdown'])}")
        
        print(f"\n🎯 RISK INTERPRETATION:")
        risk_pct = abs(worst_case_var_99) / Config.DEFAULT_PORTFOLIO_VALUE
        print(f"• On 99% of days, losses should not exceed {format_percentage(risk_pct)}")
        print(f"• In extreme scenarios (1% of days), average losses could be {format_currency(worst_case_cvar_99)}")
        
        if risk_pct > 0.05:
            print(f"⚠️  HIGH RISK: Daily VaR exceeds 5% of portfolio value")
        elif risk_pct > 0.02:
            print(f"⚡ MODERATE RISK: Daily VaR is between 2-5% of portfolio value")
        else:
            print(f"✅ LOW RISK: Daily VaR is below 2% of portfolio value")
        
        print(f"\n📈 PORTFOLIO COMPOSITION:")
        for ticker, weight in portfolio.weights.items():
            allocation = weight * Config.DEFAULT_PORTFOLIO_VALUE
            print(f"  • {ticker}: {format_percentage(weight)} ({format_currency(allocation)})")
        
        print(f"\n✅ Demo analysis completed successfully!")
        print(f"📝 This demonstrates the full Monte Carlo risk analysis workflow")
        print(f"🔄 Run with real data when Yahoo Finance API is available")
        
        # Show sample return statistics
        print(f"\n📊 SAMPLE ASSET PERFORMANCE (Simulated):")
        for ticker in Config.DEFAULT_ETFS:
            asset_returns = returns[ticker]
            annual_return = asset_returns.mean() * 252
            annual_vol = asset_returns.std() * np.sqrt(252)
            sharpe = annual_return / annual_vol if annual_vol > 0 else 0
            print(f"  • {ticker}: Return={format_percentage(annual_return)}, Vol={format_percentage(annual_vol)}, Sharpe={sharpe:.2f}")
        
        print(f"\n🎉 Portfolio Risk Analysis Demo Complete!")
        
    except Exception as e:
        logger.error(f"Error in demo execution: {e}")
        print(f"\n❌ Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)