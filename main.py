import pandas as pd
import numpy as np
import requests
from sklearn.linear_model import LinearRegression
from datetime import datetime
import yfinance as yf
import matplotlib.pyplot as plt
import io
import base64

def fetch_stock_data_yfinance(symbol, history_years):
    history_years = max(1, min(history_years, 20))
    stock = yf.Ticker(symbol)
    period_str = f"{history_years + 5}y"
    hist = stock.history(period=period_str)
    if hist.empty:
        return None, None
    hist = hist.resample('Y').last()
    hist['Year'] = hist.index.year
    hist = hist.tail(history_years)
    return hist['Year'].values.reshape(-1, 1), hist['Close'].values

def fetch_mutual_fund_nav(scheme_code, years=7):
    years = max(1, min(years, 20))
    url = f"https://api.mfapi.in/mf/{scheme_code}"
    resp = requests.get(url)
    data = resp.json()
    nav_data = data.get('data', [])
    current_year = datetime.now().year
    years_list, navs = [], []
    for entry in nav_data:
        try:
            dt = datetime.strptime(entry['date'], "%d-%m-%Y")
        except:
            continue
        if dt.year >= current_year - years:
            years_list.append(dt.year)
            navs.append(float(entry['nav']))
    if not years_list:
        return None, None
    return np.array(years_list).reshape(-1, 1), np.array(navs)

def fetch_nsc_rate():
    return 0.077  # Current NSC rate (7.7%)

def fetch_fd_rate():
    return 0.067  # Current FD rate (6.7%)

def fetch_real_estate_index(city='India', years=7):
    years = max(1, min(years, 20))
    current_year = datetime.now().year
    values, years_list = [], []
    for y in range(current_year - years, current_year + 1):
        price_index = 100 + 7*(y - (current_year - years))
        years_list.append(y)
        values.append(price_index)
    return np.array(years_list).reshape(-1, 1), np.array(values)

INVESTMENT_OPTIONS = {
    'Low': [
        {'name': 'Fixed Deposits (FDs)'},
        {'name': 'National Savings Certificate (NSC)'}
    ],
    'Medium': [
        {'name': 'Mutual Funds (Balanced/Hybrid)', 'scheme_code': '120503'},
        {'name': 'Real Estate'}
    ],
    'High': [
        {'name': 'Direct Equities (Stocks)', 'symbol': 'HDFCBANK.NS'},
        {'name': 'Equity Mutual Funds (Small/Mid-Cap)', 'scheme_code': '118550'}
    ]
}

def determine_risk_profile(age, horizon, risk_willing):
    if risk_willing.lower() == 'yes':
        if age < 35 or horizon >= 5:
            return 'High'
        else:
            return 'Medium'
    else:
        if age > 50 or horizon < 3:
            return 'Low'
        else:
            return 'Medium'

def recommend_investment_options(risk_profile):
    return [x['name'] for x in INVESTMENT_OPTIONS[risk_profile]]

def predict_future_value(years, values, future_year):
    model = LinearRegression()
    model.fit(years, values)
    return model.predict(np.array([[future_year]]))[0]

def future_value_sip(monthly_investment, annual_rate, years):
    monthly_rate = annual_rate / 12
    months = years * 12
    fv = monthly_investment * ((1 + monthly_rate) ** months - 1) / monthly_rate
    return fv

def calculate_average_annual_return(option, horizon):
    curr_year = datetime.now().year
    target_year = curr_year + horizon
    if option == 'Fixed Deposits (FDs)':
        return fetch_fd_rate()
    elif option == 'National Savings Certificate (NSC)':
        return fetch_nsc_rate()
    elif option == 'Direct Equities (Stocks)':
        years_, prices = fetch_stock_data_yfinance('HDFCBANK.NS', max(7, horizon))
        if years_ is None:
            return None
        years_flat = years_.flatten()
        years_diff = years_flat[-1] - years_flat[0]
        if years_diff == 0:
            years_diff = 1
        start_price = prices[0]
        end_price = prices[-1]
        cagr = (end_price / start_price) ** (1 / years_diff) - 1
        return cagr
    elif option == 'Mutual Funds (Balanced/Hybrid)':
        years_, navs = fetch_mutual_fund_nav('120503', max(7, horizon))
        if years_ is None:
            return None
        years_flat = years_.flatten()
        years_diff = years_flat[-1] - years_flat[0]
        if years_diff == 0:
            years_diff = 1
        start_nav = navs[0]
        end_nav = navs[-1]
        cagr = (end_nav / start_nav) ** (1 / years_diff) - 1
        return cagr
    elif option == 'Equity Mutual Funds (Small/Mid-Cap)':
        years_, navs = fetch_mutual_fund_nav('118550', max(7, horizon))
        if years_ is None:
            return None
        years_flat = years_.flatten()
        years_diff = years_flat[-1] - years_flat[0]
        if years_diff == 0:
            years_diff = 1
        start_nav = navs[0]
        end_nav = navs[-1]
        cagr = (end_nav / start_nav) ** (1 / years_diff) - 1
        return cagr
    elif option == 'Real Estate':
        years_, idx = fetch_real_estate_index('India', max(7, horizon))
        if years_ is None:
            return None
        years_flat = years_.flatten()
        years_diff = years_flat[-1] - years_flat[0]
        if years_diff == 0:
            years_diff = 1
        start_idx = idx[0]
        end_idx = idx[-1]
        cagr = (end_idx / start_idx) ** (1 / years_diff) - 1
        return cagr
    return None

def calculate_prediction(option, monthly_amount, horizon):
    horizon = max(1, min(horizon, 20))
    average_annual_return = calculate_average_annual_return(option, horizon)
    if average_annual_return is None:
        if option == 'Fixed Deposits (FDs)':
            average_annual_return = fetch_fd_rate()
        elif option == 'National Savings Certificate (NSC)':
            average_annual_return = fetch_nsc_rate()
        else:
            average_annual_return = 0.08

    fv = future_value_sip(monthly_amount, average_annual_return, horizon)
    return f"After investing ₹{monthly_amount:.2f} monthly for {horizon} years with an estimated annual return of {average_annual_return*100:.2f}%, your investment value will be approximately: ₹{fv:,.2f}"

def investment_growth_curve(option, monthly_amount, horizon):
    horizon = max(1, min(horizon, 20))
    curr_year = datetime.now().year
    months = horizon * 12
    average_annual_return = calculate_average_annual_return(option, horizon)
    if average_annual_return is None:
        if option == 'Fixed Deposits (FDs)':
            average_annual_return = fetch_fd_rate()
        elif option == 'National Savings Certificate (NSC)':
            average_annual_return = fetch_nsc_rate()
        else:
            average_annual_return = 0.08
    monthly_rate = average_annual_return / 12

    values = []
    for n in range(months + 1):
        fv_n = monthly_amount * ((1 + monthly_rate) ** n - 1) / monthly_rate
        values.append(fv_n)
    years = np.array([curr_year + (m / 12) for m in range(months + 1)])
    return years, values

def plot_investment_growth(years, values, option):
    plt.figure(figsize=(9, 4))
    plt.plot(years, values, marker='o', linestyle='-', color='#1976d2')
    plt.title(f'Projected Growth: {option}')
    plt.xlabel('Year')
    plt.ylabel('Investment Value (₹)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)

    img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    return img_base64

def calculate_prediction_with_chart(option, monthly_amount, horizon):
    pred_text = calculate_prediction(option, monthly_amount, horizon)
    years, values = investment_growth_curve(option, monthly_amount, horizon)
    if years is None or values is None:
        return pred_text, None
    img_base64 = plot_investment_growth(years, values, option)
    return pred_text, img_base64
