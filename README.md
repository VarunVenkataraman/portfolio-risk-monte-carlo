# Portfolio Risk Monte Carlo Simulation

A professional-grade quantitative finance project implementing Monte Carlo simulation for portfolio risk modeling, featuring Value-at-Risk (VaR) and Conditional VaR (CVaR) calculations, comprehensive backtesting, and stress testing methodologies.

## 🎯 Project Overview

This project demonstrates advanced quantitative risk management techniques used in institutional finance, combining statistical modeling, Monte Carlo simulation, and rigorous backtesting methodologies. Built with professional Python development practices, it serves as a comprehensive portfolio risk analysis framework.

### Key Features

- **Monte Carlo Simulation Engine**: 100,000+ simulations with multivariate normal and t-distributions
- **Risk Metrics**: VaR and CVaR calculations at multiple confidence levels (90%, 95%, 99%)
- **Correlation Modeling**: Cholesky decomposition for correlated asset returns
- **Backtesting Framework**: Kupiec POF test and Christoffersen independence testing
- **Stress Testing**: Historical scenarios and sensitivity analysis
- **Professional Architecture**: Modular design with comprehensive documentation

### ETF Universe

The default portfolio consists of seven ETFs representing major asset classes:
- **SPY**: S&P 500 (US Large Cap Equity)
- **EFA**: MSCI EAFE (Developed International Equity) 
- **EEM**: MSCI Emerging Markets (Emerging Market Equity)
- **TLT**: 20+ Year Treasury Bond (Long-term US Treasuries)
- **LQD**: Investment Grade Corporate Bond (IG Credit)
- **GLD**: Gold Trust (Commodities/Gold)
- **VNQ**: Real Estate Investment Trust (REITs)

## 🏗️ Project Structure

```
portfolio-risk-monte-carlo/
├── src/                          # Source code modules
│   ├── data/                     # Data loading and preprocessing
│   │   ├── __init__.py
│   │   └── loader.py            # Financial data acquisition
│   ├── portfolio/               # Portfolio construction and analysis
│   │   ├── __init__.py
│   │   └── construction.py      # Portfolio management utilities
│   ├── simulation/              # Monte Carlo simulation engine
│   │   ├── __init__.py
│   │   └── monte_carlo.py       # Core simulation functionality
│   ├── risk/                    # Risk metrics and calculations
│   │   ├── __init__.py
│   │   └── metrics.py           # VaR, CVaR, and risk-adjusted metrics
│   ├── backtesting/            # Model validation and backtesting
│   │   ├── __init__.py
│   │   └── validation.py        # Statistical tests and validation
│   └── utils/                   # Configuration and utilities
│       ├── __init__.py
│       └── config.py            # Project configuration
├── notebooks/                   # Jupyter notebooks for analysis
├── tests/                       # Unit tests
├── data/                        # Data storage
│   ├── raw/                     # Raw market data
│   └── processed/               # Cleaned and processed data
├── docs/                        # Additional documentation
├── requirements.txt             # Python dependencies
└── README.md                    # Project documentation
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8+ (recommended 3.9+)
- Jupyter Notebook/Lab
- Git

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/portfolio-risk-monte-carlo.git
   cd portfolio-risk-monte-carlo
   ```

2. **Create virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\\Scripts\\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify installation:**
   ```bash
   python -c "import numpy, pandas, yfinance, scipy; print('Dependencies installed successfully')"
   ```

### Quick Start Example

```python
from src.data.loader import DataLoader
from src.portfolio.construction import Portfolio
from src.simulation.monte_carlo import MonteCarloSimulator
from src.risk.metrics import RiskMetrics

# Load data
loader = DataLoader()
prices, returns = loader.get_portfolio_data(years_back=5)

# Create portfolio
portfolio = Portfolio(
    tickers=['SPY', 'EFA', 'EEM', 'TLT', 'LQD', 'GLD', 'VNQ'],
    notional_value=100000
)

# Run Monte Carlo simulation
simulator = MonteCarloSimulator(
    returns=returns[portfolio.tickers],
    portfolio_weights=portfolio.get_weights_array()
)

# Generate 100,000 simulations
pnl_simulations = simulator.simulate_portfolio_pnl(
    portfolio_value=100000,
    n_simulations=100000
)

# Calculate risk metrics
var_95 = RiskMetrics.value_at_risk(pnl_simulations, confidence_level=0.95)
cvar_95 = RiskMetrics.conditional_value_at_risk(pnl_simulations, confidence_level=0.95)

print(f"95% VaR: ${var_95:,.2f}")
print(f"95% CVaR: ${cvar_95:,.2f}")
```

## 📊 Core Modules

### Data Loading (`src/data/loader.py`)
- **DataLoader**: Fetches ETF price data from Yahoo Finance
- Handles data caching and validation
- Calculates simple and log returns
- Data quality checks and cleaning

### Portfolio Construction (`src/portfolio/construction.py`)
- **Portfolio**: Portfolio management with flexible weighting
- **PortfolioOptimizer**: Risk parity and minimum variance optimization
- **PortfolioAnalyzer**: Performance attribution and reporting
- Correlation and covariance matrix calculations

### Monte Carlo Simulation (`src/simulation/monte_carlo.py`)
- **MonteCarloSimulator**: Core simulation engine
- Multivariate normal and t-distribution sampling
- Cholesky decomposition for correlation structure
- Stress testing with modified market conditions
- Confidence interval calculations

### Risk Metrics (`src/risk/metrics.py`)
- **RiskMetrics**: VaR and CVaR calculations (historical, parametric, Monte Carlo)
- **RiskDecomposition**: Component VaR and marginal risk analysis
- **RiskReporting**: Comprehensive risk reporting utilities
- Maximum drawdown and risk-adjusted return metrics

### Backtesting (`src/backtesting/validation.py`)
- **VaRBacktest**: Statistical validation of VaR models
- Kupiec Proportion of Failures (POF) test
- Christoffersen independence and joint tests
- Rolling backtest analysis
- **StressTesting**: Scenario analysis and sensitivity testing

## 🧪 Statistical Methods

### Monte Carlo Implementation
- **Correlation Structure**: Cholesky decomposition ensures proper correlation between assets
- **Distribution Choice**: Support for multivariate normal and t-distributions
- **Variance Reduction**: Antithetic variates and moment matching available
- **Numerical Stability**: Regularization for non-positive definite covariance matrices

### Risk Measures
- **Value-at-Risk (VaR)**: Quantile-based loss estimation at 90%, 95%, 99% confidence
- **Conditional VaR (CVaR)**: Expected loss beyond VaR threshold (coherent risk measure)
- **Component VaR**: Risk contribution decomposition by asset
- **Maximum Drawdown**: Peak-to-trough decline analysis

### Backtesting Framework
- **Kupiec POF Test**: Tests if violation rate matches expected frequency
- **Christoffersen Tests**: Independence and joint coverage testing
- **Rolling Analysis**: Time-varying model performance assessment
- **Statistical Significance**: Chi-square tests with proper degrees of freedom

## 📈 Usage Examples

### Basic Risk Analysis
```python
# Generate risk report
from src.risk.metrics import RiskReporting

report = RiskReporting.generate_risk_report(
    returns=portfolio_returns,
    portfolio_value=100000
)

print(f"Annual Sharpe Ratio: {report['risk_metrics']['risk_adjusted_returns']['sharpe_ratio']:.3f}")
print(f"Maximum Drawdown: {report['risk_metrics']['drawdown_analysis']['max_drawdown']:.3%}")
```

### Stress Testing
```python
# Define stress scenarios
stress_scenarios = {
    'market_crash': {
        'vol_multiplier': 2.0,
        'return_shock': -0.05
    },
    'correlation_breakdown': {
        'correlation_multiplier': 0.1
    }
}

# Run stress tests
stress_results = simulator.run_stress_test(
    stress_scenarios=stress_scenarios,
    portfolio_value=100000
)
```

### VaR Model Backtesting
```python
from src.backtesting.validation import VaRBacktest

# Historical VaR forecasts (example)
var_forecasts = pd.Series(...)  # Your VaR model forecasts

# Create backtest
backtest = VaRBacktest(
    returns=portfolio_returns,
    var_forecasts=var_forecasts,
    confidence_level=0.95
)

# Run statistical tests
pof_results = backtest.kupiec_pof_test()
independence_results = backtest.christoffersen_independence_test()

print(f"POF Test p-value: {pof_results['p_value']:.4f}")
print(f"Model Adequate: {not pof_results['reject_null']}")
```

## 🔬 Testing

Run the test suite to ensure all components work correctly:

```bash
# Run all tests
python -m pytest tests/ -v

# Run with coverage
python -m pytest tests/ --cov=src --cov-report=html

# Run specific test module
python -m pytest tests/test_monte_carlo.py -v
```

## 📋 Dependencies

### Core Libraries
- **numpy**: Numerical computing and array operations
- **pandas**: Data manipulation and time series analysis
- **scipy**: Statistical functions and optimization
- **yfinance**: Financial data acquisition

### Visualization
- **matplotlib**: Static plotting and charts
- **seaborn**: Statistical data visualization
- **plotly**: Interactive visualizations

### Development Tools
- **pytest**: Testing framework
- **black**: Code formatting
- **flake8**: Linting and style checking
- **mypy**: Static type checking

## 🎓 Educational Value

This project demonstrates several key quantitative finance concepts:

1. **Portfolio Theory**: Modern portfolio theory implementation with risk-return optimization
2. **Risk Management**: Industry-standard VaR and CVaR methodologies
3. **Statistical Modeling**: Multivariate distributions and correlation modeling
4. **Model Validation**: Rigorous backtesting with statistical significance tests
5. **Stress Testing**: Scenario analysis and sensitivity testing frameworks

## 📊 Performance Considerations

- **Simulation Speed**: Vectorized numpy operations for 100k+ simulations
- **Memory Management**: Efficient array operations and chunked processing
- **Numerical Stability**: Regularization techniques for covariance matrices
- **Caching**: Data caching to avoid repeated API calls

## 🤝 Contributing

Contributions are welcome! Areas for enhancement:

- [ ] GPU acceleration for Monte Carlo simulations
- [ ] Additional distribution models (e.g., copulas)
- [ ] Real-time portfolio monitoring
- [ ] Options and derivatives risk modeling
- [ ] Machine learning-based VaR models

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📞 Contact

**Varun Sharma**  
Email: [your.email@example.com]  
LinkedIn: [linkedin.com/in/yourprofile]  
GitHub: [github.com/yourusername]

## 🙏 Acknowledgments

- **Yahoo Finance API** for providing market data
- **Quantitative Finance Literature** for methodological frameworks
- **Open Source Community** for excellent Python libraries

---

*This project serves as a demonstration of quantitative finance expertise and professional Python development practices. It is designed for educational and portfolio purposes.*