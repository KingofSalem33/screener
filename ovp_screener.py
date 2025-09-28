import pandas as pd
import numpy as np
from scipy.stats import norm
import requests
import time


# === LOAD TICKERS FROM FILE ===
tickers_df = pd.read_csv("tickers.csv")
TICKERS = tickers_df['Ticker'].tolist() if 'Ticker' in tickers_df.columns else tickers_df.iloc[:, 0].tolist()
# ROIC.ai demo limitations - only AAPL is free, other tickers require Professional plan
# For demo purposes, let's just use AAPL multiple times or add more demo-supported tickers
DEMO_TICKERS = ['AAPL']  # Only AAPL is freely available
TICKERS = DEMO_TICKERS

# ROIC.ai API configuration
ROIC_BASE_URL = "https://api.roic.ai/v2"


# === NORMALIZATION ===
def normalize(series: pd.Series, lower_better=False) -> pd.Series:
    """Normalize series to percentile scores"""
    if len(series) < 2:
        # If only one value, return 50th percentile
        return pd.Series([50.0] * len(series), index=series.index)

    mean = series.mean()
    std = series.std(ddof=0)

    # Avoid division by zero
    if std == 0:
        return pd.Series([50.0] * len(series), index=series.index)

    z = (series - mean) / std
    p = norm.cdf(z)
    return (1 - p) * 100 if lower_better else p * 100

# === ROIC.AI API FUNCTIONS ===
def get_roic_company_profile(ticker):
    """Get company profile data from ROIC.ai"""
    try:
        url = f"{ROIC_BASE_URL}/company/profile/{ticker}"
        response = requests.get(url)

        if response.status_code != 200:
            print(f"Failed to get profile for {ticker}: {response.status_code}")
            return None

        data = response.json()

        # Check for error message indicating Professional plan required
        if isinstance(data, dict) and 'error' in data:
            print(f"ROIC.ai error for {ticker}: {data['error']}")
            return None

        return data[0] if data else None

    except Exception as e:
        print(f"ROIC profile error for {ticker}: {e}")
        return None

def get_roic_income_statement(ticker):
    """Get income statement data from ROIC.ai"""
    try:
        url = f"{ROIC_BASE_URL}/fundamental/income-statement/{ticker}"
        response = requests.get(url)

        if response.status_code != 200:
            print(f"Failed to get income statement for {ticker}: {response.status_code}")
            return None

        data = response.json()
        return data if data else None

    except Exception as e:
        print(f"ROIC income statement error for {ticker}: {e}")
        return None

def get_roic_balance_sheet(ticker):
    """Get balance sheet data from ROIC.ai"""
    try:
        url = f"{ROIC_BASE_URL}/fundamental/balance-sheet/{ticker}"
        response = requests.get(url)

        if response.status_code != 200:
            print(f"Failed to get balance sheet for {ticker}: {response.status_code}")
            return None

        data = response.json()
        return data if data else None

    except Exception as e:
        print(f"ROIC balance sheet error for {ticker}: {e}")
        return None

def get_roic_cash_flow(ticker):
    """Get cash flow data from ROIC.ai"""
    try:
        url = f"{ROIC_BASE_URL}/fundamental/cash-flow/{ticker}"
        response = requests.get(url)

        if response.status_code != 200:
            print(f"Failed to get cash flow for {ticker}: {response.status_code}")
            return None

        data = response.json()
        return data if data else None

    except Exception as e:
        print(f"ROIC cash flow error for {ticker}: {e}")
        return None

def get_stock_data_with_retries(ticker, max_retries=3):
    """Get stock data from ROIC.ai with retry logic"""
    for attempt in range(max_retries):
        try:
            # Get all financial data from ROIC.ai
            profile = get_roic_company_profile(ticker)
            income_statements = get_roic_income_statement(ticker)
            balance_sheets = get_roic_balance_sheet(ticker)
            cash_flows = get_roic_cash_flow(ticker)

            if not profile or not income_statements:
                print(f"No data for {ticker}, retry {attempt + 1}")
                time.sleep(1)
                continue

            return {
                'profile': profile,
                'income_statements': income_statements,
                'balance_sheets': balance_sheets,
                'cash_flows': cash_flows
            }

        except Exception as e:
            print(f"Attempt {attempt + 1} failed for {ticker}: {e}")
            if attempt < max_retries - 1:
                time.sleep(1)
            else:
                print(f"All attempts failed for {ticker}")
                return None

    return None

# === MAIN SCREENING FUNCTION ===
def main():
    records = []

    for ticker in TICKERS:
        try:
            print(f"Fetching {ticker} from ROIC.ai...")

            # Get data using ROIC.ai API
            stock_data = get_stock_data_with_retries(ticker)

            if not stock_data:
                print(f"Skipping {ticker} - no data available")
                continue

            profile = stock_data['profile']
            income_statements = stock_data['income_statements']
            balance_sheets = stock_data['balance_sheets']
            cash_flows = stock_data['cash_flows']

            # Check if we have enough data for growth calculations
            if len(income_statements) < 2:
                print(f"Insufficient historical data for {ticker}")
                continue

            # Calculate growth rates using ROIC.ai data structure
            current_year = income_statements[0]  # Most recent year
            prev_year = income_statements[1]     # Previous year

            # Revenue growth
            current_revenue = current_year.get('is_sales_revenue_turnover', 0)
            prev_revenue = prev_year.get('is_sales_revenue_turnover', 0)
            rev_growth = 0
            if prev_revenue != 0:
                rev_growth = ((current_revenue - prev_revenue) / abs(prev_revenue)) * 100

            # EPS growth (already calculated in ROIC.ai data)
            current_eps = current_year.get('eps', 0)
            prev_eps = prev_year.get('eps', 0)
            eps_growth = 0
            if prev_eps != 0:
                eps_growth = ((current_eps - prev_eps) / abs(prev_eps)) * 100

            # Operating margin (already calculated in ROIC.ai data)
            op_margin = current_year.get('oper_margin', 0)

            # Financial ratios
            de = np.nan
            if balance_sheets and len(balance_sheets) > 0:
                de = balance_sheets[0].get('net_debt_to_shrhldr_eqy', np.nan)

            # Calculate P/E, P/S, P/B ratios
            current_price = profile.get('price', 0)
            pe = np.nan
            ps = np.nan
            pb = np.nan

            if current_price > 0:
                current_eps_val = current_year.get('eps', 0)
                if current_eps_val > 0:
                    pe = current_price / current_eps_val

                # Calculate P/S ratio properly using shares outstanding
                if balance_sheets and len(balance_sheets) > 0:
                    shares_out = balance_sheets[0].get('bs_sh_out', 0)
                    if shares_out > 0 and current_revenue > 0:
                        revenue_per_share = current_revenue / shares_out
                        ps = current_price / revenue_per_share

                book_value_per_share = 0
                if balance_sheets and len(balance_sheets) > 0:
                    book_value = balance_sheets[0].get('bs_total_equity', 0)
                    shares_out = balance_sheets[0].get('bs_sh_out', 1)
                    if shares_out > 0:
                        book_value_per_share = book_value / shares_out
                        pb = current_price / book_value_per_share if book_value_per_share > 0 else np.nan

            # Dividend yield
            div_yield = profile.get('dividend_yield', 0) * 100

            # Cash flow metrics
            net_income = current_year.get('is_net_income', 0)
            cash_flow_ops = 0
            free_cash_flow = 0

            if cash_flows and len(cash_flows) > 0:
                cash_flow_ops = cash_flows[0].get('cf_cash_from_oper', 0)
                free_cash_flow = cash_flows[0].get('cf_free_cash_flow', 0)

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

        # Rate limiting for ROIC.ai API
        time.sleep(0.5)

    df = pd.DataFrame(records)
    print(f"Created dataframe with {len(df)} records")
    print(f"Columns: {df.columns.tolist()}")

    print(f"\nNote: ROIC.ai API limitations - Only AAPL is available for demo/free access.")
    print(f"Other tickers require a Professional plan subscription.")

    if df.empty:
        print("No data collected. Exiting.")
        return

    # Show raw data for debugging
    print("\nRaw data:")
    print(df.to_string())

    # Check for NaN values before dropping
    print(f"\nNaN counts per column:")
    print(df.isnull().sum())

    # Fill NaN values with reasonable defaults instead of dropping rows
    df = df.fillna({'Debt_to_Equity': 0})  # Assume no debt if missing

    if len(df) < 2:
        print(f"\nWarning: With only {len(df)} stock(s), normalization scores will all be 50.0 (median).")
        print("The OVP scoring system works best with multiple stocks for comparison.")

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
    print("\nTop Stocks by Overall Value Potential (OVP):")
    print(top.to_string(index=False))

    # Save to file
    top.to_csv("top_stocks_by_OVP.csv", index=False)
    print("\nResults saved to 'top_stocks_by_OVP.csv'")

if __name__ == '__main__':
    main()
