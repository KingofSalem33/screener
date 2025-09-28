import pandas as pd
import numpy as np
from scipy.stats import norm
import yfinance as yf

# === LOAD TICKERS FROM FILE ===
tickers_df = pd.read_csv("C:/Users/coryh/OneDrive/Cory - Personal/Ticker - Sheet1.csv")
TICKERS = tickers_df['Ticker'].dropna().tolist()


# === NORMALIZATION ===
def normalize(series: pd.Series, lower_better=False) -> pd.Series:
    mean = series.mean()
    std = series.std(ddof=0)
    z = (series - mean) / std
    p = norm.cdf(z)
    return (1 - p) * 100 if lower_better else p * 100

# === MAIN SCREENING FUNCTION ===
def main():
    records = []

    for ticker in TICKERS:
        try:
            print(f"Fetching {ticker}...")
            stock = yf.Ticker(ticker)
            info = stock.info

            # Get historical financials for annual data
            financials = stock.financials  # Annual income statement
            balance_sheet = stock.balance_sheet  # Annual balance sheet
            cashflow = stock.cashflow  # Annual cash flow

            # Calculate annual growth rates from historical data
            eps_growth = 0
            rev_growth = 0
            if not financials.empty and len(financials.columns) >= 2:
                # Get most recent 2 years for growth calculation
                recent_year = financials.columns[0]
                prev_year = financials.columns[1]

                # Revenue growth (year-over-year)
                if 'Total Revenue' in financials.index:
                    recent_rev = financials.loc['Total Revenue', recent_year]
                    prev_rev = financials.loc['Total Revenue', prev_year]
                    if prev_rev != 0 and not pd.isna(recent_rev) and not pd.isna(prev_rev):
                        rev_growth = ((recent_rev - prev_rev) / abs(prev_rev)) * 100

                # EPS growth calculation using Net Income and shares outstanding
                if 'Net Income' in financials.index:
                    recent_ni = financials.loc['Net Income', recent_year]
                    prev_ni = financials.loc['Net Income', prev_year]
                    shares_outstanding = info.get('sharesOutstanding', 1)
                    if prev_ni != 0 and not pd.isna(recent_ni) and not pd.isna(prev_ni) and shares_outstanding:
                        recent_eps = recent_ni / shares_outstanding
                        prev_eps = prev_ni / shares_outstanding
                        if prev_eps != 0:
                            eps_growth = ((recent_eps - prev_eps) / abs(prev_eps)) * 100

            # Operating margin from most recent year
            op_margin = 0
            if not financials.empty and 'Total Revenue' in financials.index and 'Operating Income' in financials.index:
                recent_year = financials.columns[0]
                revenue = financials.loc['Total Revenue', recent_year]
                operating_income = financials.loc['Operating Income', recent_year]
                if revenue != 0 and not pd.isna(revenue) and not pd.isna(operating_income):
                    op_margin = (operating_income / revenue) * 100

            # Keep current ratios from yfinance info (these are already annual/trailing)
            de = info.get('debtToEquity') or np.nan
            pe = info.get('trailingPE') or np.nan
            ps = info.get('priceToSalesTrailing12Months') or np.nan
            pb = info.get('priceToBook') or np.nan
            div_yield = (info.get('dividendYield') or 0) * 100

            # Get cash flow data from yfinance annual data
            net_income = 0
            cash_flow_ops = 0
            free_cash_flow = 0

            if not financials.empty and 'Net Income' in financials.index:
                net_income = financials.loc['Net Income', financials.columns[0]]

            if not cashflow.empty:
                recent_year = cashflow.columns[0]
                if 'Operating Cash Flow' in cashflow.index:
                    cash_flow_ops = cashflow.loc['Operating Cash Flow', recent_year]
                if 'Free Cash Flow' in cashflow.index:
                    free_cash_flow = cashflow.loc['Free Cash Flow', recent_year]
                elif 'Operating Cash Flow' in cashflow.index and 'Capital Expenditure' in cashflow.index:
                    # Calculate FCF if not directly available
                    ocf = cashflow.loc['Operating Cash Flow', recent_year]
                    capex = cashflow.loc['Capital Expenditure', recent_year]
                    if not pd.isna(ocf) and not pd.isna(capex):
                        free_cash_flow = ocf + capex  # capex is usually negative

            records.append({
                'Ticker': ticker,
                'EPS_Growth': eps_growth,
                'Revenue_Growth': rev_growth,
                'Operating_Margin': op_margin,
                'Debt_to_Equity': de,
                'Net_Income': net_income,
                'Operating_Cash_Flow': cash_flow_ops,
                'Free_Cash_Flow': free_cash_flow,
                'P/E': pe,
                'P/S': ps,
                'P/B': pb,
                'Dividend_Yield': div_yield
            })

        except Exception as e:
            print(f"Error with {ticker}: {e}")

    df = pd.DataFrame(records).dropna()

    # === NORMALIZE ===
    df['EPS_Growth_N'] = normalize(df['EPS_Growth'])
    df['Revenue_Growth_N'] = normalize(df['Revenue_Growth'])
    df['Operating_Margin_N'] = normalize(df['Operating_Margin'])
    df['Debt_to_Equity_N'] = normalize(df['Debt_to_Equity'], lower_better=True)
    df['Net_Income_N'] = normalize(df['Net_Income'])
    df['Operating_Cash_Flow_N'] = normalize(df['Operating_Cash_Flow'])
    df['Free_Cash_Flow_N'] = normalize(df['Free_Cash_Flow'])

    df['P/E_N'] = normalize(df['P/E'], lower_better=True)
    df['P/S_N'] = normalize(df['P/S'], lower_better=True)
    df['P/B_N'] = normalize(df['P/B'], lower_better=True)
    df['Dividend_Yield_N'] = normalize(df['Dividend_Yield'])

    # === FPR AND NVR ===
    df['FPR'] = (
        df['EPS_Growth_N'] * 0.25 +
        df['Revenue_Growth_N'] * 0.20 +
        df['Operating_Margin_N'] * 0.15 +
        df['Debt_to_Equity_N'] * 0.10 +
        df['Net_Income_N'] * 0.10 +
        df['Operating_Cash_Flow_N'] * 0.10 +
        df['Free_Cash_Flow_N'] * 0.10
    )

    df['NVR'] = (
        df['P/E_N'] * 0.35 +
        df['P/S_N'] * 0.25 +
        df['P/B_N'] * 0.25 +
        df['Dividend_Yield_N'] * 0.15
    )

    # === HARMONIC MEAN OVP ===
    df['OVP'] = 2 * (df['FPR'] * df['NVR']) / (df['FPR'] + df['NVR'])

    # === OUTPUT ===
    top = df.sort_values('OVP', ascending=False)[['Ticker', 'FPR', 'NVR', 'OVP']]
    print("\n🔝 Top Stocks by Overall Value Potential (OVP):")
    print(top.to_string(index=False))

    # Save to file
    top.to_csv("top_stocks_by_OVP.csv", index=False)
    print("\n✅ Results saved to 'top_stocks_by_OVP.csv'")

if __name__ == '__main__':
    main()
