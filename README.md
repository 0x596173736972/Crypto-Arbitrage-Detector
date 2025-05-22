# Crypto Arbitrage Trading System

A real-time arbitrage detection system for cryptocurrency markets.

## Features

- Real-time price monitoring across Binance and Kraken exchanges
- Automated arbitrage opportunity detection with profit calculations
- Interactive Streamlit dashboard for visualization and configuration
- Historical tracking of arbitrage opportunities
- Risk assessment and profitability analysis

## Requirements

- Python 3.8+
- Dependencies listed in `requirements.txt`

## Installation

1. Clone the repository
2. Create a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Copy `.env.example` to `.env` and add your exchange API keys:
   ```
   cp .env.example .env
   ```

## Usage

Run the Streamlit dashboard:

```
streamlit run app.py
```

This will start the application and open it in your web browser.

## Configuration

1. Use the sidebar to select trading pairs and exchanges
2. Adjust the profit threshold to filter arbitrage opportunities
3. Configure advanced settings in the Settings tab

## Project Structure

- `app.py` - Main Streamlit application
- `config.py` - Configuration settings
- `components/` - UI components for the dashboard
- `services/` - Core business logic and exchange connectivity

## Disclaimer

This software is for educational purposes only. Cryptocurrency trading involves significant risk. Always conduct your own research before making any trading decisions.

## License

MIT