#!/usr/bin/env python3

# WEI Market Overview Dashboard

Bloomberg-style market monitor for Equities, Futures, FX, and Crypto.  
Built with Python, Matplotlib, Pandas, and Yahoo Finance.

## Features
- Bloomberg-style layout
- 2D intraday spark charts (5d/1m)
- YTD performance sparkline
- Sections: Equities, MAG7, Futures, Rates, Energy, Metals, Ags, FX, Crypto

## Example Output
![Market Overview Screenshot](screenshot.png)

## How to Run
```bash
pip install -r requirements.txt
python market_overview.py



matplotlib==3.8.4
numpy==1.26.4
pandas==2.2.2
yfinance==0.2.43
python-dateutil==2.9.0.post0
pytz==2024.1


