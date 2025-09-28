# 📊 Value Potential Screener

A Node.js tool that computes a stock's **Overall Value Potential (OVP)** using the **Financial Performance Rating (FPR)** and **Normalized Valuation Rating (NVR)** framework.

Built with [yahoo-finance2](https://www.npmjs.com/package/yahoo-finance2).

---

## 🚀 Features
- Fetches **latest quarterly & TTM financials** from Yahoo Finance.
- Computes:
  - **FPR** (Financial Performance Rating)
    - EPS growth YoY (quarterly)
    - Revenue growth YoY (quarterly)
    - Operating margin (TTM)
    - Debt-to-Equity (MRQ)
    - ROIC (TTM, approx via NOPAT / Invested Capital)
  - **NVR** (Normalized Valuation Rating)
    - Price-to-Earnings (P/E, trailing; invalid if EPS ≤ 0)
    - Price-to-Sales (P/S, TTM)
    - Price-to-Book (P/B, MRQ)
    - EV/EBITDA (TTM, invalid if EBITDA ≤ 0)
    - Dividend Yield (%)
- **Invalid metrics** are automatically **dropped** and weights redistributed.
- **Clips negative normalized scores to 0** for consistency.
- Uses **harmonic mean** to combine FPR & NVR into the final OVP.
- Outputs strict **JSON** schema with scores and a **ranking**.

---

## 📦 Installation

```bash
# Clone repo or copy screener.js + README.md
npm init -y
npm i yahoo-finance2
```

## 🖥️ Usage
Run the screener with a comma-separated list of tickers:

```bash
node screener.js --tickers ENPH,FSLR,SEDG,NEE,RUN
```

## Example Output (truncated)
```json
{
  "as_of_date": "2025-09-27",
  "universe": ["ENPH","FSLR","SEDG","NEE","RUN"],
  "method": {
    "normalization": "percentile",
    "notes": "Normalized within provided universe; lower-is-better metrics inverted; negatives clipped to 0; NVR auto-drops invalid valuation metrics."
  },
  "results": [
    {
      "ticker": "ENPH",
      "fpr_inputs": {
        "eps_growth_yoy_pct": 37.1,
        "revenue_growth_yoy_pct": 12.3,
        "operating_margin_pct": 12.8,
        "de_ratio": 0.36,
        "roic_pct": 8.1
      },
      "nvr_inputs": {
        "pe": { "value": 45.1, "valid": true },
        "ps": { "value": 10.0, "valid": true },
        "pb": { "value": 5.2, "valid": true },
        "ev_ebitda": { "value": 27.4, "valid": true },
        "div_yield_pct": { "value": null, "valid": false }
      },
      "scores": {
        "FPR": 70.2,
        "NVR": 55.6,
        "OVP": 62.1
      }
    }
  ],
  "ranking": [
    { "ticker": "ENPH", "OVP": 62.1, "rank": 1 },
    { "ticker": "FSLR", "OVP": 59.8, "rank": 2 }
  ]
}
```

## 🧮 Scoring Methodology

### Financial Performance Rating (FPR)
```
FPR = Σ(wi × XN,i*) / Σwi valid
```

- **EPS Growth YoY** (30%)
- **Revenue Growth YoY** (25%)
- **Operating Margin** (20%)
- **(100 – D/E)** (15%)
- **ROIC** (10%)

### Normalized Valuation Rating (NVR)
```
NVR = Σ(wi × XN,i*) / Σwi valid
```

- **P/E** (30%)
- **P/S** (20%)
- **P/B** (15%)
- **EV/EBITDA** (25%)
- **Dividend Yield** (10%)

Invalid metrics (e.g., negative earnings → P/E invalid) are dropped and weights are redistributed.

### Overall Value Potential (OVP)
```
OVP = 2×(FPR×NVR) / (FPR+NVR)
```

Harmonic mean penalizes imbalance (e.g., high fundamentals but poor valuation, or vice versa).

## 📊 Interpretation
- **80–100** → Exceptional overall potential
- **60–79** → Strong balance of fundamentals & valuation
- **40–59** → Moderate potential
- **20–39** → Weak fundamentals or stretched valuation
- **0–19** → Highly speculative or unattractive

## ⚠️ Notes
- Yahoo Finance sometimes lags SEC filings. Cross-check if results look odd.
- **FPR** is fundamental health, **NVR** is valuation attractiveness.
- **OVP** is not a buy/sell signal — it's a screening tool.
- Negative metrics (EPS, margins, ROIC) → clipped to 0.
- Missing/invalid valuation metrics → auto-dropped, weights redistributed.

## 📚 References
- [Yahoo Finance API (yahoo-finance2)](https://www.npmjs.com/package/yahoo-finance2)
- SEC EDGAR Filings (source of Yahoo data)
- Harmonic mean method for balancing multi-factor scores

## 🛠️ Future Improvements
- Sector-adjusted weight tuning
- Forward estimate support (FWD EPS, EBITDA)
- Add CSV / table output
- Deploy as a web dashboard or cloud API