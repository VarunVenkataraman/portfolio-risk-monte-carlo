"""
Main execution script for Portfolio Risk Monte Carlo Simulation.

This script demonstrates the complete workflow from data loading to risk analysis.
Run this script to generate a comprehensive risk report for the default ETF portfolio.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import logging
from datetime import datetime

# Import project modules
from src.data.loader import DataLoader
from src.portfolio.construction import Portfolio, PortfolioAnalyzer
from src.simulation.monte_carlo import MonteCarloSimulator
from src.risk.metrics import RiskReporting
from src.backtesting.validation import StressTesting
from src.utils.config import Config, format_currency, format_percentage

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """
    Main execution function - demonstrates complete portfolio risk analysis workflow.
    """
    print("=" * 80)
    print("PORTFOLIO RISK MONTE CARLO SIMULATION")
    print("=" * 80)
    print(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Portfolio Value: {format_currency(Config.DEFAULT_PORTFOLIO_VALUE)}")
    print(f"ETF Universe: {', '.join(Config.DEFAULT_ETFS)}")
    print("=" * 80)
    
    try:
        # Step 1: Load Market Data
        print("\n📊 STEP 1: LOADING MARKET DATA")
        print("-" * 40)
        
        loader = DataLoader(cache_dir="data/raw")
        prices, returns = loader.get_portfolio_data(
            tickers=Config.DEFAULT_ETFS,
            years_back=Config.DEFAULT_YEARS_BACK,
            return_type=Config.DEFAULT_RETURN_TYPE
        )
        
        print(f"✓ Loaded {len(returns)} days of historical data")
        print(f"✓ Date range: {returns.index.min().date()} to {returns.index.max().date()}")
        
        # Data validation
        validation = loader.validate_data(returns)
        print(f"✓ Data validation complete - {validation['shape'][1]} assets, {validation['shape'][0]} observations")
        
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
        n_simulations = Config.DEFAULT_SIMULATIONS
        print(f"🔄 Running {n_simulations:,} Monte Carlo simulations...")
        
        pnl_simulations = simulator.simulate_portfolio_pnl(
            portfolio_value=Config.DEFAULT_PORTFOLIO_VALUE,
            n_simulations=n_simulations,
            time_horizon=Config.DEFAULT_TIME_HORIZON
        )
        
        print(f"✓ Simulations complete")
        
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
        
        # Historical crisis scenarios
        if len(returns) > 252 * 3:  # If we have enough data
            crisis_results = stress_tester.historical_scenarios(Config.CRISIS_PERIODS)
            
            if crisis_results:
                print("\n🏛️ HISTORICAL CRISIS SCENARIOS:")
                for crisis_name, results in crisis_results.items():
                    print(f"\n📍 {crisis_name.replace('_', ' ').title()} ({results['period']}):")
                    print(f"   • Worst Day: {format_percentage(results['worst_day'])}")
                    print(f"   • Total Return: {format_percentage(results['total_return'])}")
                    print(f"   • Max Drawdown: {format_percentage(results['max_drawdown'])}")
        
        # Step 6: Summary and Recommendations
        print("\n" + "=" * 80)
        print("📋 EXECUTIVE SUMMARY")
        print("=" * 80)
        
        worst_case_var_99 = mc_summary['var_99']
        worst_case_cvar_99 = mc_summary['cvar_99']
        
        print(f"Portfolio Value: {format_currency(Config.DEFAULT_PORTFOLIO_VALUE)}")
        print(f"Time Horizon: {Config.DEFAULT_TIME_HORIZON} day(s)")
        print(f"Confidence Level: 99%")
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
        
        print(f"\n✅ Analysis completed successfully!")
        print(f"📁 Results can be found in the 'results' directory")
        
        # Save results to file (optional)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_summary = {
            'timestamp': timestamp,
            'portfolio_value': Config.DEFAULT_PORTFOLIO_VALUE,
            'monte_carlo_metrics': mc_summary,
            'risk_report': risk_report,
            'stress_test_results': stress_results
        }
        
        # Create results directory
        os.makedirs('results', exist_ok=True)
        
        # Save to CSV for easy viewing
        summary_df = pd.DataFrame({
            'Metric': ['99% VaR', '99% CVaR', '95% VaR', '95% CVaR', 'Sharpe Ratio', 'Max Drawdown'],
            'Value': [
                format_currency(mc_summary['var_99']),
                format_currency(mc_summary['cvar_99']),
                format_currency(mc_summary['var_95']),
                format_currency(mc_summary['cvar_95']),
                f"{risk_report['risk_metrics']['risk_adjusted_returns']['sharpe_ratio']:.3f}",
                format_percentage(risk_report['risk_metrics']['drawdown_analysis']['max_drawdown'])
            ]
        })
        
        summary_df.to_csv(f'results/risk_summary_{timestamp}.csv', index=False)
        print(f"📄 Summary saved to: results/risk_summary_{timestamp}.csv")
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        print(f"\n❌ Error: {e}")
        print("Please check the logs and ensure all dependencies are installed.")
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)