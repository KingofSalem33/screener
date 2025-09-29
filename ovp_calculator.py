import pandas as pd
import numpy as np
import requests
import json
from datetime import datetime
from scipy import stats

# Financial Modeling Prep API configuration
FMP_API_KEY = "eiVcr0EhatmV9aGUeSGTaSTvnlC4ihdq"  # Legacy key - get new free key from FMP website
FMP_BASE_URL = "https://financialmodelingprep.com/api/v3"

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

def normalize_to_percentile(value, mean, std, lower_is_better=False, metric_type=None):
    """Convert raw value to 0-100 percentile using industry benchmarks

    Args:
        value: Raw metric value
        mean: Industry average
        std: Industry standard deviation
        lower_is_better: Whether lower values are better (for valuation ratios)
        metric_type: Type of metric for special handling ('earnings', 'growth', 'margin', 'ratio')
    """
    if value is None or pd.isna(value):
        return None

    # Special handling for negative earnings-related metrics
    if metric_type in ['earnings', 'growth'] and value < 0:
        # Map negative earnings/growth to 5th percentile (poor performance)
        return 5.0

    # Special handling for negative margins
    if metric_type == 'margin' and value < 0:
        # Map negative margins to 10th percentile (very poor performance)
        return 10.0

    # Special handling for invalid ratios (negative earnings making P/E invalid)
    if metric_type == 'ratio' and (value <= 0 or value > 1000):  # Extreme or invalid ratios
        if lower_is_better:
            # For valuation ratios, extreme values indicate poor value (high percentile = bad)
            return 5.0  # Map to 5th percentile (good value score since lower is better)
        else:
            return 10.0  # Map to 10th percentile for other ratios

    # Normal calculation for valid values
    # Calculate z-score
    z_score = (value - mean) / std

    # Convert to percentile (0-100)
    percentile = stats.norm.cdf(z_score) * 100

    # Invert for "lower is better" metrics
    if lower_is_better:
        percentile = 100 - percentile

    # Clip to 0-100 range
    return max(0, min(100, percentile))

def get_fmp_data(ticker):
    """Fetch comprehensive financial data from Financial Modeling Prep"""
    try:
        # Get company profile
        profile_url = f"{FMP_BASE_URL}/profile/{ticker}?apikey={FMP_API_KEY}"
        profile_response = requests.get(profile_url)

        if profile_response.status_code == 200:
            profile_json = profile_response.json()
            if isinstance(profile_json, dict) and 'Error Message' in profile_json:
                print(f"FMP Error: {profile_json['Error Message']}")
                print("Please get a new free API key from: https://site.financialmodelingprep.com/developer/docs")
                return None
            profile_data = profile_json[0] if profile_json else None
        else:
            profile_data = None

        # Get income statement (quarterly data - latest 8 quarters)
        income_url = f"{FMP_BASE_URL}/income-statement/{ticker}?period=quarter&limit=8&apikey={FMP_API_KEY}"
        income_response = requests.get(income_url)
        income_data = income_response.json() if income_response.status_code == 200 else None

        # Get balance sheet (quarterly data - latest 4 quarters)
        balance_url = f"{FMP_BASE_URL}/balance-sheet-statement/{ticker}?period=quarter&limit=4&apikey={FMP_API_KEY}"
        balance_response = requests.get(balance_url)
        balance_data = balance_response.json() if balance_response.status_code == 200 else None

        # Get cash flow (quarterly data - latest 4 quarters)
        cashflow_url = f"{FMP_BASE_URL}/cash-flow-statement/{ticker}?period=quarter&limit=4&apikey={FMP_API_KEY}"
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
    """Extract and calculate all required financial metrics from FMP data"""
    if not data or not data['profile'] or not data['income']:
        return None

    profile = data['profile']
    income_statements = data['income']
    balance_sheets = data['balance'] if data['balance'] else []
    cashflows = data['cashflow'] if data['cashflow'] else []

    # Get latest quarterly data (FMP returns most recent first)
    latest_quarter = income_statements[0]
    # For YoY comparison, find same quarter from previous year (4 quarters back)
    prev_year_quarter = income_statements[4] if len(income_statements) > 4 else None

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
        current_rev = latest_quarter.get('revenue', 0)
        prev_rev = prev_year_quarter.get('revenue', 0)
        if prev_rev != 0:
            metrics['revenue_growth'] = ((current_rev - prev_rev) / abs(prev_rev)) * 100
        else:
            metrics['revenue_growth'] = None
    else:
        metrics['revenue_growth'] = None

    # 3. Operating Margin (%) - Calculate from latest quarter
    revenue = latest_quarter.get('revenue', 0)
    operating_income = latest_quarter.get('operatingIncome', 0)
    if revenue > 0 and operating_income is not None:
        metrics['operating_margin'] = (operating_income / revenue) * 100
    else:
        metrics['operating_margin'] = None

    # 4. Debt-to-Equity Ratio - Calculate from most recent balance sheet
    if balance_sheets:
        total_debt = balance_sheets[0].get('totalDebt', 0)
        total_equity = balance_sheets[0].get('totalStockholdersEquity', 0)
        if total_equity > 0:
            metrics['debt_to_equity'] = total_debt / total_equity
        else:
            metrics['debt_to_equity'] = None
    else:
        metrics['debt_to_equity'] = None

    # 5. ROIC (%) - Calculate from available data
    if balance_sheets and operating_income:
        total_equity = balance_sheets[0].get('totalStockholdersEquity', 0)
        total_debt = balance_sheets[0].get('totalDebt', 0)
        invested_capital = total_equity + total_debt
        # Estimate NOPAT (assume 25% tax rate)
        nopat = operating_income * 0.75
        if invested_capital > 0:
            metrics['roic'] = (nopat / invested_capital) * 100
        else:
            metrics['roic'] = None
    else:
        metrics['roic'] = None

    # NVR Metrics
    current_price = profile.get('price', 0)
    market_cap = profile.get('mktCap', 0)

    # 6. P/E Ratio (TTM) - Use profile data or calculate
    pe_ratio = profile.get('pe', None)
    if not pe_ratio and current_price > 0:
        current_eps = latest_quarter.get('eps', 0)
        if current_eps > 0:
            pe_ratio = current_price / current_eps
    metrics['pe_ratio'] = pe_ratio

    # 7. P/S Ratio (TTM) - Calculate using market cap and TTM revenue
    # Sum last 4 quarters for TTM revenue
    ttm_revenue = sum([q.get('revenue', 0) for q in income_statements[:4]])
    if market_cap > 0 and ttm_revenue > 0:
        metrics['ps_ratio'] = market_cap / ttm_revenue
    else:
        metrics['ps_ratio'] = None

    # 8. P/B Ratio (MRQ) - Use profile data
    metrics['pb_ratio'] = profile.get('priceToBook', None)

    # 9. EV/EBITDA (TTM)
    # Calculate Enterprise Value
    net_debt = balance_sheets[0].get('netDebt', 0) if balance_sheets else 0
    enterprise_value = market_cap + net_debt

    # Sum last 4 quarters for TTM EBITDA
    ttm_ebitda = sum([q.get('ebitda', 0) for q in income_statements[:4]])
    if ttm_ebitda > 0 and enterprise_value > 0:
        metrics['ev_ebitda'] = enterprise_value / ttm_ebitda
    else:
        metrics['ev_ebitda'] = None

    # 10. Dividend Yield (%)
    metrics['dividend_yield'] = profile.get('lastDiv', 0) / current_price * 100 if current_price > 0 else 0

    return metrics

def calculate_ovp(ticker):
    """Main function to calculate OVP score for a given ticker"""
    print(f"\n=== OVERALL VALUE POTENTIAL CALCULATOR ===")
    print(f"Analyzing: {ticker}")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d')}")

    # Step 1: Fetch latest earnings data
    print("\nStep 1: Fetching latest quarterly earnings data from FMP...")
    data = get_fmp_data(ticker)

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

    # Normalize FPR components with proper metric type handling
    eps_n = normalize_to_percentile(metrics['eps_growth'],
                                   INDUSTRY_BENCHMARKS['eps_growth']['mean'],
                                   INDUSTRY_BENCHMARKS['eps_growth']['std'],
                                   metric_type='growth')

    rev_n = normalize_to_percentile(metrics['revenue_growth'],
                                   INDUSTRY_BENCHMARKS['revenue_growth']['mean'],
                                   INDUSTRY_BENCHMARKS['revenue_growth']['std'],
                                   metric_type='growth')

    opmargin_n = normalize_to_percentile(metrics['operating_margin'],
                                        INDUSTRY_BENCHMARKS['operating_margin']['mean'],
                                        INDUSTRY_BENCHMARKS['operating_margin']['std'],
                                        metric_type='margin')

    de_n = normalize_to_percentile(metrics['debt_to_equity'],
                                  INDUSTRY_BENCHMARKS['debt_to_equity']['mean'],
                                  INDUSTRY_BENCHMARKS['debt_to_equity']['std'],
                                  lower_is_better=True)

    roic_n = normalize_to_percentile(metrics['roic'],
                                    INDUSTRY_BENCHMARKS['roic']['mean'],
                                    INDUSTRY_BENCHMARKS['roic']['std'],
                                    metric_type='earnings')

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

    # Normalize NVR components with proper metric type handling
    pe_n = normalize_to_percentile(metrics['pe_ratio'],
                                  INDUSTRY_BENCHMARKS['pe_ratio']['mean'],
                                  INDUSTRY_BENCHMARKS['pe_ratio']['std'],
                                  lower_is_better=True,
                                  metric_type='ratio')

    ps_n = normalize_to_percentile(metrics['ps_ratio'],
                                  INDUSTRY_BENCHMARKS['ps_ratio']['mean'],
                                  INDUSTRY_BENCHMARKS['ps_ratio']['std'],
                                  lower_is_better=True,
                                  metric_type='ratio')

    pb_n = normalize_to_percentile(metrics['pb_ratio'],
                                  INDUSTRY_BENCHMARKS['pb_ratio']['mean'],
                                  INDUSTRY_BENCHMARKS['pb_ratio']['std'],
                                  lower_is_better=True,
                                  metric_type='ratio')

    evebitda_n = normalize_to_percentile(metrics['ev_ebitda'],
                                        INDUSTRY_BENCHMARKS['ev_ebitda']['mean'],
                                        INDUSTRY_BENCHMARKS['ev_ebitda']['std'],
                                        lower_is_better=True,
                                        metric_type='ratio')

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

    # Check for negative earnings handling
    negative_metrics = []
    if metrics['eps_growth'] and metrics['eps_growth'] < 0:
        negative_metrics.append("EPS Growth")
    if metrics['revenue_growth'] and metrics['revenue_growth'] < 0:
        negative_metrics.append("Revenue Growth")
    if metrics['operating_margin'] and metrics['operating_margin'] < 0:
        negative_metrics.append("Operating Margin")
    if metrics['pe_ratio'] and (metrics['pe_ratio'] <= 0 or metrics['pe_ratio'] > 1000):
        negative_metrics.append("P/E Ratio")

    if negative_metrics:
        print(f"\nNote: Negative/invalid metrics detected: {', '.join(negative_metrics)}")
        print("These have been mapped to 5th-10th percentile to reflect poor performance without distortion.")

    # Show raw metrics for transparency
    print(f"\n=== RAW FINANCIAL METRICS ===")
    print(f"EPS Growth (YoY): {metrics['eps_growth']:.2f}%" if metrics['eps_growth'] is not None else "EPS Growth: N/A")
    print(f"Revenue Growth (YoY): {metrics['revenue_growth']:.2f}%" if metrics['revenue_growth'] is not None else "Revenue Growth: N/A")
    print(f"Operating Margin: {metrics['operating_margin']:.2f}%" if metrics['operating_margin'] is not None else "Operating Margin: N/A")
    print(f"Debt-to-Equity: {metrics['debt_to_equity']:.2f}" if metrics['debt_to_equity'] is not None else "Debt-to-Equity: N/A")
    print(f"ROIC: {metrics['roic']:.2f}%" if metrics['roic'] is not None else "ROIC: N/A")
    print(f"P/E Ratio: {metrics['pe_ratio']:.2f}" if metrics['pe_ratio'] is not None else "P/E Ratio: N/A")
    print(f"P/S Ratio: {metrics['ps_ratio']:.2f}" if metrics['ps_ratio'] is not None else "P/S Ratio: N/A")
    print(f"P/B Ratio: {metrics['pb_ratio']:.2f}" if metrics['pb_ratio'] is not None else "P/B Ratio: N/A")
    print(f"EV/EBITDA: {metrics['ev_ebitda']:.2f}" if metrics['ev_ebitda'] is not None else "EV/EBITDA: N/A")
    print(f"Dividend Yield: {metrics['dividend_yield']:.2f}%" if metrics['dividend_yield'] is not None else "Dividend Yield: 0.00%")

    return {
        'ticker': ticker,
        'ovp': ovp,
        'fpr': fpr,
        'nvr': nvr,
        'metrics': metrics,
        'interpretation': interpretation
    }

if __name__ == "__main__":
    # NOTE: You need to get a new free API key from FMP
    # Visit: https://site.financialmodelingprep.com/developer/docs
    # Sign up for free account and replace the API key above

    # Example usage
    ticker = "AAPL"
    print("NOTE: If you see API errors, please get a new free API key from:")
    print("https://site.financialmodelingprep.com/developer/docs")
    print("Then update the FMP_API_KEY variable in this file.\n")

    result = calculate_ovp(ticker)