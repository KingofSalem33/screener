import pandas as pd
import numpy as np
import requests
import json
from datetime import datetime
from scipy import stats

# ROIC.ai API configuration
ROIC_BASE_URL = "https://api.roic.ai/v2"

# Industry averages and standard deviations for normalization (Technology sector baseline)
INDUSTRY_BENCHMARKS = {
    'eps_growth': {'mean': 15.2, 'std': 25.8},
    'revenue_growth': {'mean': 12.5, 'std': 18.3},
    'operating_margin': {'mean': 22.1, 'std': 12.4},
    'debt_to_equity': {'mean': 45.2, 'std': 35.7},
    'roic': {'mean': 18.5, 'std': 14.2},
    'pe_ratio': {'mean': 28.4, 'std': 15.6},
    'ps_ratio': {'mean': 8.2, 'std': 6.8},
    'pb_ratio': {'mean': 4.8, 'std': 3.2},
    'ev_ebitda': {'mean': 22.1, 'std': 12.8},
    'dividend_yield': {'mean': 1.8, 'std': 2.1}
}

def normalize_to_percentile(value, mean, std, lower_is_better=False):
    """Convert raw value to 0-100 percentile using industry benchmarks"""
    if value is None or pd.isna(value):
        return None

    # Calculate z-score
    z_score = (value - mean) / std

    # Convert to percentile (0-100)
    percentile = stats.norm.cdf(z_score) * 100

    # Invert for "lower is better" metrics
    if lower_is_better:
        percentile = 100 - percentile

    # Clip to 0-100 range
    return max(0, min(100, percentile))

def get_roic_data(ticker):
    """Fetch comprehensive financial data from ROIC.ai"""
    try:
        # Get company profile
        profile_url = f"{ROIC_BASE_URL}/company/profile/{ticker}"
        profile_response = requests.get(profile_url)
        profile_data = profile_response.json()[0] if profile_response.status_code == 200 else None

        # Get income statement (quarterly data)
        income_url = f"{ROIC_BASE_URL}/fundamental/income-statement/{ticker}"
        income_response = requests.get(income_url)
        income_data = income_response.json() if income_response.status_code == 200 else None

        # Get balance sheet
        balance_url = f"{ROIC_BASE_URL}/fundamental/balance-sheet/{ticker}"
        balance_response = requests.get(balance_url)
        balance_data = balance_response.json() if balance_response.status_code == 200 else None

        # Get cash flow
        cashflow_url = f"{ROIC_BASE_URL}/fundamental/cash-flow/{ticker}"
        cashflow_response = requests.get(cashflow_url)
        cashflow_data = cashflow_response.json() if cashflow_response.status_code == 200 else None

        return {
            'profile': profile_data,
            'income': income_data,
            'balance': balance_data,
            'cashflow': cashflow_data
        }
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return None

def calculate_metrics(data):
    """Extract and calculate all required financial metrics"""
    if not data or not data['profile'] or not data['income']:
        return None

    profile = data['profile']
    income_statements = data['income']
    balance_sheets = data['balance'] if data['balance'] else []
    cashflows = data['cashflow'] if data['cashflow'] else []

    # Get latest quarterly data
    latest_quarter = income_statements[0]
    prev_year_quarter = income_statements[1] if len(income_statements) > 1 else None

    metrics = {}

    # FPR Metrics
    # 1. EPS Growth (YoY %) - Latest quarter vs same quarter prior year
    if prev_year_quarter:
        current_eps = latest_quarter.get('eps', 0)
        prev_eps = prev_year_quarter.get('eps', 0)
        if prev_eps != 0:
            metrics['eps_growth'] = ((current_eps - prev_eps) / abs(prev_eps)) * 100
        else:
            metrics['eps_growth'] = None
    else:
        metrics['eps_growth'] = None

    # 2. Revenue Growth (YoY %)
    if prev_year_quarter:
        current_rev = latest_quarter.get('is_sales_revenue_turnover', 0)
        prev_rev = prev_year_quarter.get('is_sales_revenue_turnover', 0)
        if prev_rev != 0:
            metrics['revenue_growth'] = ((current_rev - prev_rev) / abs(prev_rev)) * 100
        else:
            metrics['revenue_growth'] = None
    else:
        metrics['revenue_growth'] = None

    # 3. Operating Margin (%) - TTM from latest data
    metrics['operating_margin'] = latest_quarter.get('oper_margin', None)

    # 4. Debt-to-Equity Ratio - MRQ
    if balance_sheets:
        metrics['debt_to_equity'] = balance_sheets[0].get('net_debt_to_shrhldr_eqy', None)
    else:
        metrics['debt_to_equity'] = None

    # 5. ROIC (%) - Estimate from available data
    if balance_sheets and latest_quarter.get('is_oper_income'):
        total_equity = balance_sheets[0].get('bs_total_equity', 0)
        total_debt = balance_sheets[0].get('net_debt', 0)
        invested_capital = total_equity + total_debt
        nopat = latest_quarter.get('is_oper_income', 0) * 0.79  # Assume 21% tax rate
        if invested_capital > 0:
            metrics['roic'] = (nopat / invested_capital) * 100
        else:
            metrics['roic'] = None
    else:
        metrics['roic'] = None

    # NVR Metrics
    current_price = profile.get('price', 0)

    # 6. P/E Ratio (TTM)
    current_eps = latest_quarter.get('eps', 0)
    if current_eps > 0 and current_price > 0:
        metrics['pe_ratio'] = current_price / current_eps
    else:
        metrics['pe_ratio'] = None

    # 7. P/S Ratio (TTM)
    if balance_sheets and current_price > 0:
        shares_out = balance_sheets[0].get('bs_sh_out', 0)
        revenue = latest_quarter.get('is_sales_revenue_turnover', 0)
        if shares_out > 0 and revenue > 0:
            revenue_per_share = revenue / shares_out
            metrics['ps_ratio'] = current_price / revenue_per_share
        else:
            metrics['ps_ratio'] = None
    else:
        metrics['ps_ratio'] = None

    # 8. P/B Ratio (MRQ)
    if balance_sheets and current_price > 0:
        book_value = balance_sheets[0].get('bs_total_equity', 0)
        shares_out = balance_sheets[0].get('bs_sh_out', 0)
        if shares_out > 0 and book_value > 0:
            book_value_per_share = book_value / shares_out
            metrics['pb_ratio'] = current_price / book_value_per_share
        else:
            metrics['pb_ratio'] = None
    else:
        metrics['pb_ratio'] = None

    # 9. EV/EBITDA (TTM)
    market_cap = current_price * balance_sheets[0].get('bs_sh_out', 0) if balance_sheets else 0
    net_debt = balance_sheets[0].get('net_debt', 0) if balance_sheets else 0
    enterprise_value = market_cap + net_debt
    ebitda = latest_quarter.get('ebitda', 0)
    if ebitda > 0 and enterprise_value > 0:
        metrics['ev_ebitda'] = enterprise_value / ebitda
    else:
        metrics['ev_ebitda'] = None

    # 10. Dividend Yield (%)
    metrics['dividend_yield'] = profile.get('dividend_yield', 0) * 100 if profile.get('dividend_yield') else 0

    return metrics

def calculate_ovp(ticker):
    """Main function to calculate OVP score for a given ticker"""
    print(f"\n=== OVERALL VALUE POTENTIAL CALCULATOR ===")
    print(f"Analyzing: {ticker}")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d')}")

    # Step 1: Fetch latest earnings data
    print("\nStep 1: Fetching latest quarterly earnings data...")
    data = get_roic_data(ticker)

    if not data:
        print(f"Error: Could not fetch data for {ticker}")
        return None

    # Extract metrics
    metrics = calculate_metrics(data)
    if not metrics:
        print(f"Error: Could not calculate metrics for {ticker}")
        return None

    print("Data retrieved successfully")

    # Step 2: Normalize metrics and calculate FPR and NVR
    print("\nStep 2: Normalizing metrics and calculating ratings...")

    # Normalize FPR components
    eps_n = normalize_to_percentile(metrics['eps_growth'],
                                   INDUSTRY_BENCHMARKS['eps_growth']['mean'],
                                   INDUSTRY_BENCHMARKS['eps_growth']['std'])

    rev_n = normalize_to_percentile(metrics['revenue_growth'],
                                   INDUSTRY_BENCHMARKS['revenue_growth']['mean'],
                                   INDUSTRY_BENCHMARKS['revenue_growth']['std'])

    opmargin_n = normalize_to_percentile(metrics['operating_margin'],
                                        INDUSTRY_BENCHMARKS['operating_margin']['mean'],
                                        INDUSTRY_BENCHMARKS['operating_margin']['std'])

    de_n = normalize_to_percentile(metrics['debt_to_equity'],
                                  INDUSTRY_BENCHMARKS['debt_to_equity']['mean'],
                                  INDUSTRY_BENCHMARKS['debt_to_equity']['std'],
                                  lower_is_better=True)

    roic_n = normalize_to_percentile(metrics['roic'],
                                    INDUSTRY_BENCHMARKS['roic']['mean'],
                                    INDUSTRY_BENCHMARKS['roic']['std'])

    # Calculate FPR
    fpr_components = [
        (eps_n, 0.30),
        (rev_n, 0.25),
        (opmargin_n, 0.20),
        (de_n, 0.15),
        (roic_n, 0.10)
    ]

    fpr = 0
    total_weight = 0
    for score, weight in fpr_components:
        if score is not None:
            fpr += score * weight
            total_weight += weight

    if total_weight > 0:
        fpr = fpr / total_weight * sum([w for _, w in fpr_components])
    else:
        fpr = 0

    # Normalize NVR components
    pe_n = normalize_to_percentile(metrics['pe_ratio'],
                                  INDUSTRY_BENCHMARKS['pe_ratio']['mean'],
                                  INDUSTRY_BENCHMARKS['pe_ratio']['std'],
                                  lower_is_better=True)

    ps_n = normalize_to_percentile(metrics['ps_ratio'],
                                  INDUSTRY_BENCHMARKS['ps_ratio']['mean'],
                                  INDUSTRY_BENCHMARKS['ps_ratio']['std'],
                                  lower_is_better=True)

    pb_n = normalize_to_percentile(metrics['pb_ratio'],
                                  INDUSTRY_BENCHMARKS['pb_ratio']['mean'],
                                  INDUSTRY_BENCHMARKS['pb_ratio']['std'],
                                  lower_is_better=True)

    evebitda_n = normalize_to_percentile(metrics['ev_ebitda'],
                                        INDUSTRY_BENCHMARKS['ev_ebitda']['mean'],
                                        INDUSTRY_BENCHMARKS['ev_ebitda']['std'],
                                        lower_is_better=True)

    div_n = normalize_to_percentile(metrics['dividend_yield'],
                                   INDUSTRY_BENCHMARKS['dividend_yield']['mean'],
                                   INDUSTRY_BENCHMARKS['dividend_yield']['std'])

    # Calculate NVR
    nvr_components = [
        (pe_n, 0.30),
        (ps_n, 0.20),
        (pb_n, 0.15),
        (evebitda_n, 0.25),
        (div_n, 0.10)
    ]

    nvr = 0
    total_weight = 0
    for score, weight in nvr_components:
        if score is not None:
            nvr += score * weight
            total_weight += weight

    if total_weight > 0:
        nvr = nvr / total_weight * sum([w for _, w in nvr_components])
    else:
        nvr = 0

    print("Ratings calculated successfully")

    # Step 3: Calculate OVP using harmonic mean
    print("\nStep 3: Calculating Overall Value Potential (OVP)...")

    if fpr > 0 and nvr > 0:
        ovp = 2 * (fpr * nvr) / (fpr + nvr)
    else:
        ovp = 0

    # Display results
    print(f"\n=== RESULTS FOR {ticker} ===")
    print(f"Financial Performance Rating (FPR): {fpr:.1f}")
    print(f"Normalized Valuation Rating (NVR): {nvr:.1f}")
    print(f"Overall Value Potential (OVP): {ovp:.1f}")

    # Interpretation
    if ovp >= 80:
        interpretation = "Exceptional investment potential - strong financials and attractive valuation"
    elif ovp >= 60:
        interpretation = "Good investment potential - solid overall profile"
    elif ovp >= 40:
        interpretation = "Moderate investment potential - mixed signals"
    elif ovp >= 20:
        interpretation = "Weak investment potential - concerning metrics"
    else:
        interpretation = "Poor investment potential - significant challenges"

    print(f"Interpretation: {interpretation}")

    # Show raw metrics for transparency
    print(f"\n=== RAW FINANCIAL METRICS ===")
    print(f"EPS Growth (YoY): {metrics['eps_growth']:.2f}%" if metrics['eps_growth'] else "EPS Growth: N/A")
    print(f"Revenue Growth (YoY): {metrics['revenue_growth']:.2f}%" if metrics['revenue_growth'] else "Revenue Growth: N/A")
    print(f"Operating Margin: {metrics['operating_margin']:.2f}%" if metrics['operating_margin'] else "Operating Margin: N/A")
    print(f"Debt-to-Equity: {metrics['debt_to_equity']:.2f}" if metrics['debt_to_equity'] else "Debt-to-Equity: N/A")
    print(f"ROIC: {metrics['roic']:.2f}%" if metrics['roic'] else "ROIC: N/A")
    print(f"P/E Ratio: {metrics['pe_ratio']:.2f}" if metrics['pe_ratio'] else "P/E Ratio: N/A")
    print(f"P/S Ratio: {metrics['ps_ratio']:.2f}" if metrics['ps_ratio'] else "P/S Ratio: N/A")
    print(f"P/B Ratio: {metrics['pb_ratio']:.2f}" if metrics['pb_ratio'] else "P/B Ratio: N/A")
    print(f"EV/EBITDA: {metrics['ev_ebitda']:.2f}" if metrics['ev_ebitda'] else "EV/EBITDA: N/A")
    print(f"Dividend Yield: {metrics['dividend_yield']:.2f}%" if metrics['dividend_yield'] else "Dividend Yield: 0.00%")

    return {
        'ticker': ticker,
        'ovp': ovp,
        'fpr': fpr,
        'nvr': nvr,
        'metrics': metrics,
        'interpretation': interpretation
    }

if __name__ == "__main__":
    # Example usage
    ticker = "AAPL"
    result = calculate_ovp(ticker)