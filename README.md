# WEI-Mkt-Overview-
WEI Mkt Overview 
# Bloomberg-Style Market Dashboard

A Python-powered market overview dashboard inspired by Bloomberg WEI.  
It visualizes live global market data using Yahoo Finance, with features like:

- 📊 **Equities, Futures, FX, Crypto, Rates, Commodities**
- 🌍 Regional breakdowns (Americas, EMEA, APAC)
- 📈 **Two-tone intraday sparks** (yesterday in white, today vs. prev close)
- 🕛 **YTD spark lines** in slim far-right columns
- ✅ Auto-aligned tables, Bloomberg-like look

---

## 📸 Screenshot

---<img width="2017" height="1297" alt="wei overview" src="https://github.com/user-attachments/assets/81919a10-457e-4b0b-a942-4df2897a85fb" />



## ⚙️ Installation

Clone the repo:

```bash
git clone https://github.com/bruce1095/WEI-Mkt-Overview-.git
cd WEI-Mkt-Overview-


python3 -m venv venv
source venv/bin/activate   # on Mac/Linux
venv\Scripts\activate      # on Windows


pip install -r requirements.txt


python market_overview.py

The script will:
	•	Fetch 1y/1h historical data
	•	Fetch 5d/1m intraday for sparks
	•	Build Bloomberg-style overview tables
	•	Display the dashboard in a Matplotlib figure

⸻

📦 Requirements

The key Python packages are:
	•	numpy
	•	pandas
	•	yfinance
	•	matplotlib
	•	zoneinfo (standard in Python 3.9+)

All pinned in requirements.txt.

⸻

✨ Example Sections
	•	EQUITIES: S&P 500, Dow, Nasdaq, Russell, VIX
	•	MAG 7: AAPL, MSFT, NVDA, AMZN, META, TSLA, GOOGL
	•	FUTURES: ES, NQ, YM, RTY
	•	RATES, ENERGY, METALS, AGS, FX, CRYPTO

⸻

📌 Notes
	•	Data is pulled from Yahoo Finance, aligned to local sessions.
	•	All sparks are automatically scaled to fit.
	•	Built to replicate the Bloomberg WEI monitor with open-source tools.

⸻

🔗 Author

Built by Bruce
