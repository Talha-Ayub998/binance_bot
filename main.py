import os
import json
import math
import time
import logging
import threading
import schedule
import requests
import functools
from datetime import datetime, timedelta, timezone
from pathlib import Path
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
from binance.client import Client
from binance.enums import *
from binance.exceptions import BinanceAPIException

# Load the .env file
load_dotenv()

# Create a custom logger
logger = logging.getLogger('binance_bot')
logger.setLevel(logging.INFO)


# Retrieve the environment variables
API_KEY = os.getenv("API_KEY")
API_SECRET = os.getenv("API_SECRET")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")

TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

FUTURE_API_KEY_ = os.getenv("FUTURE_API_KEY")
FUTURE_API_SECRET = os.getenv("FUTURE_API_SECRET")

# Initialize Binance Client
client = Client(API_KEY, API_SECRET, testnet=True)

client_future = Client(FUTURE_API_KEY_, FUTURE_API_SECRET, testnet=True)
# Globals
PORTFOLIO_VALUE = 100  # Example portfolio value
TOP_N_COINS = 10
# Global dictionary to track orders
orders = {}
is_paused = False  # Global flag to control scheduling

market_type_map = {}
USE_DYNAMIC_VOL_SIZING = True
TARGET_VOL = 0.45
ALLOCATION_PER_COIN_PERCENT = 0.05
MAX_POSITIONS = 20

# Decorator for rate-limiting and exponential backoff
def rate_limit_retry(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        retries = 5
        delay = 1  # Initial delay in seconds
        for attempt in range(retries):
            try:
                return func(*args, **kwargs)
            except BinanceAPIException as e:
                if e.code == -1003:  # API rate limit exceeded
                    time.sleep(delay)
                    delay *= 2  # Exponential backoff
                else:
                    raise
        raise Exception("Rate limit exceeded after retries.")
    return wrapper


def send_telegram_alert(message=None, image_path=None):
    base_url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}"
    retries = 3
    delay = 2  # Initial delay for retry mechanism

    # Send a text message if provided
    if message:
        url = f"{base_url}/sendMessage"
        payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message}

        for attempt in range(retries):
            try:
                response = requests.post(url, json=payload, timeout=10)
                if response.status_code == 200:
                    break
                else:
                    raise Exception(f"Failed to send message: {response.text}")
            except Exception as e:
                if attempt == retries - 1:
                    print_log(f"Telegram message failed after retries: {e}")
                else:
                    time.sleep(delay)
                    delay *= 2  # Exponential backoff

    # Send an image if provided
    if image_path:
        url = f"{base_url}/sendPhoto"

        for attempt in range(retries):
            try:
                with open(image_path, "rb") as image:
                    files = {"photo": image}
                    payload = {"chat_id": TELEGRAM_CHAT_ID}
                    response = requests.post(url, data=payload, files=files, timeout=10)
                    if response.status_code == 200:
                        break
                    else:
                        raise Exception(f"Failed to send image: {response.text}")
            except Exception as e:
                if attempt == retries - 1:
                    print_log(f"Telegram image failed after retries: {e}")
                else:
                    time.sleep(delay)
                    delay *= 2  # Exponential backoff)


def get_folder_logger():
    # Determine the current working directory and log file path
    folder_path = Path(os.getcwd())
    log_file = folder_path / f"{folder_path.name}.log"

    # Create a unique logger name based on the folder name
    logger_name = f"{folder_path.name}_logger"
    logger = logging.getLogger(logger_name)

    # If the logger has no handlers, configure it
    if not logger.hasHandlers():
        logger.setLevel(logging.DEBUG)

        # Create a file handler for logging
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)

        # Define and set the formatter for the handler
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)

        # Add the handler to the logger
        logger.addHandler(file_handler)

    return logger


def print_log(message):
    # Print the message to the console
    print(message)
    # Retrieve the logger and log the message at INFO level
    logger = get_folder_logger()
    logger.info(message)


@rate_limit_retry
def fetch_ohlcv(symbol, interval="1d", lookback="30 days ago UTC"):
    try:
        # Parse lookback into a start time (convert "50 days ago UTC" to an actual timestamp)
        if "day" in lookback:
            # Extract the number of days
            days = int(lookback.split()[0])
            start_time = int(
                (datetime.now(timezone.utc) -
                 timedelta(days=days)).timestamp() * 1000
            )
        else:
            # Fallback: assume lookback is a valid datetime string
            start_time = int(pd.Timestamp(lookback).timestamp() * 1000)
        # Request historical data from Binance API
        # url = "https://api.binance.us/api/v3/klines"

        url = "https://api.binance.com/api/v3/klines"
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": start_time,
        }
        response = requests.get(url, params=params)
        # Check for errors
        if response.status_code != 200:
            print_log(f"Error fetching data  from Binance: for {symbol}: {response.text}")
            return pd.DataFrame()
        # Parse response JSON
        raw_data = response.json()
        # Convert to DataFrame
        data = pd.DataFrame(raw_data, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base', 'taker_buy_quote', 'ignore'
        ])
        # Process and clean data
        data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
        data.set_index('timestamp', inplace=True)
        data = data[['open', 'high', 'low', 'close', 'volume']].astype(float)
        return data
    except Exception as e:
        print_log(f"Error in fetch_ohlcv: {e}")
        return pd.DataFrame()


def calculate_roc30(data):
    """
    Calculate the 30-day Rate of Change (ROC) for the latest available day using pct_change.
    """
    try:
        # Calculate percentage change over a 30-day period
        roc = data['close'].pct_change(periods=30) * 100
        # Return the second-to-last value (equivalent to -2 in manual calculation)
        return roc.iloc[-2]
    except Exception as e:
        print(f"Error calculating ROC30: {e}")
        return None


def calculate_vwap(data):
    if data is None:
        return "Error: Input data is None."

    required_columns = {'close', 'volume'}
    if not required_columns.issubset(data.columns):
        return f"Error: Missing required columns. Required: {required_columns}"

    try:
        vwap = (data['close'] * data['volume']).sum() / data['volume'].sum()
        return vwap
    except Exception as e:
        return f"Error calculating VWAP: {e}"



def is_strategy_active(symbol, ma_period=50, interval="1d", lookback="51 days ago UTC"):
    """Check if the symbol's closing price(up to yesterday) is above its moving average(MA)."""
    try:
        data = fetch_ohlcv(symbol, interval=interval, lookback=lookback)
        if data.empty or len(data) < ma_period + 1:
            print_log(
                f"Not enough data to calculate {ma_period}-day MA for {symbol}.")
            return False
        data = data.iloc[:-1]
        ma = data['close'].rolling(window=ma_period).mean().iloc[-1]
        return data['close'].iloc[-1] > ma
    except Exception as e:
        print_log(f"Error in is_strategy_active: {e}")
        return False


def get_bottom_10_coins_usdt():
    all_symbols = get_top_50_liquid_symbols()
    filtered_coins = []
    for symbol in all_symbols:
        try:
            data = fetch_ohlcv(symbol, interval="1d", lookback="32 days ago UTC")
            if len(data) < 32:
                continue
            roc30 = calculate_roc30(data)
            volume = (data['volume'] * data['close']).iloc[-8:-1].mean()
            if volume > 10_000_000:
                filtered_coins.append({'symbol': symbol, 'ROC30': roc30})
        except:
            continue
    filtered_coins = [c for c in filtered_coins if c['ROC30'] is not None]
    bottom_10 = sorted(filtered_coins, key=lambda x: x['ROC30'])[:TOP_N_COINS]

    print_log(f"Bottom 10 Coins: {bottom_10}")
    return bottom_10

# ---- Strategy Functions ----
@rate_limit_retry
def get_top_10_coins_usdt():
    try:
        """Get symbols of the top 10 USDT pairs by ROC30 and volume."""
        symbols = get_top_50_liquid_symbols()
        # symbols = [s['symbol'] for s in client.get_exchange_info(
        # )['symbols'] if s['quoteAsset'] == 'USDT']
        filtered_coins = []
        for symbol in symbols:
            try:
                data = fetch_ohlcv(symbol, interval="1d", lookback="32 days ago UTC")
                # Skip if data has less than 32 rows (including current date)
                if len(data) < 32:
                    print_log(
                        f"{symbol}: Insufficient data (less than 32 days including current day). Skipping...")
                    continue
                roc30 = calculate_roc30(data)
                volume = (data['volume'] * data['close']).iloc[-8:-1].mean()
                if volume and volume > 10_000_000:
                    filtered_coins.append({'symbol': symbol, 'ROC30': roc30})
                else:
                    print_log(f"{symbol}: Insufficient liquidity (volume = {volume}). Skipping...")
            except Exception as e:
                print_log(f"Error fetching data for {symbol}: {e}")
        filtered_coins = [
            coin for coin in filtered_coins if coin['ROC30'] is not None]
        # Sort by ROC30 and pick the top N symbols
        top_symbols = sorted(
            filtered_coins, key=lambda x: x['ROC30'], reverse=True)[:TOP_N_COINS]
        # top_symbols = [coin['symbol'] for coin in sorted(
        #     filtered_coins, key=lambda x: x['ROC30'], reverse=True)[:TOP_N_COINS]]
        print_log(f"Top 10 Coins: {top_symbols}")

        return top_symbols
    except Exception as e:
        print_log(f"Error in get_top_10_coins_usdt: {e}")
        return []


def get_portfolio_value():
    try:
        account_info = client_future.futures_account()
        balance = float(account_info["totalWalletBalance"])
        return balance
    except:
        return 0

def place_order(symbol, side, quantity, price=None, market_type="SPOT"):
    try:
        if market_type == "SPOT":
            return client.create_order(
                symbol=symbol, side=side,
                type=ORDER_TYPE_MARKET if price is None else ORDER_TYPE_LIMIT,
                quantity=quantity,
                price=f"{price:.8f}" if price else None,
                timeInForce=TIME_IN_FORCE_GTC if price else None
            )
        else:
            return client_future.futures_create_order(
                symbol=symbol, side=side,
                type=FUTURE_ORDER_TYPE_MARKET if price is None else FUTURE_ORDER_TYPE_LIMIT,
                quantity=quantity,
                price=f"{price:.8f}" if price else None,
                timeInForce=TIME_IN_FORCE_GTC if price else None
            )
    except Exception as e:
        send_telegram_alert(f"Order error: {e}")
        return None

def fetch_7day_avg_volume(symbol, market_type="SPOT"):
    try:
        if market_type == "FUTURES":
            klines = client_future.futures_klines(symbol=symbol, interval="1d", limit=7)
        else:
            klines = client.get_klines(symbol=symbol, interval="1d", limit=7)
        return np.mean([float(c[5]) * float(c[4]) for c in klines])
    except:
        return 0

def get_preferred_market(symbol_base):
    spot_symbol = f"{symbol_base}USDT"
    futures_symbol = f"{symbol_base}USDT"
    spot_volume = fetch_7day_avg_volume(spot_symbol, "SPOT")
    futures_volume = fetch_7day_avg_volume(futures_symbol, "FUTURES")
    if futures_volume > spot_volume:
        market_type_map[spot_symbol] = "FUTURES"
        return futures_symbol, "FUTURES"
    else:
        market_type_map[spot_symbol] = "SPOT"
        return spot_symbol, "SPOT"


def get_ma20_signals(df):
    ma20 = df['close'].rolling(window=20).mean()
    cross_above = (df['close'] > ma20) & (df['close'].shift(1) <= ma20.shift(1))
    cross_below = df['close'] < ma20
    return ma20, cross_above, cross_below


def get_top_50_liquid_symbols():
    symbols = [
        s['symbol'] for s in client.get_exchange_info()['symbols']
        if s['quoteAsset'] == 'USDT' and s['status'] == 'TRADING'
    ]
    volume_scores = {}
    for symbol in symbols:
        df = fetch_ohlcv(symbol, interval="1d", lookback="32 days ago UTC")
        if len(df) < 30: continue
        dv = df['close'] * df['volume']
        volume_scores[symbol] = dv.iloc[-30:].mean()
    return sorted(volume_scores, key=volume_scores.get, reverse=True)[:50]

def calculate_volatility_position_size(returns, window=14, target_vol=TARGET_VOL):
    rolling_vol = returns.rolling(window=window).std() * np.sqrt(252)
    latest_vol = rolling_vol.iloc[-1]
    if latest_vol <= 0 or np.isnan(latest_vol):
        raise ValueError("Invalid volatility.")
    return target_vol / latest_vol

def check_margin_health(n_warn=0.30, n_sell=0.20, n_stop=0.10):
    try:
        data = client_future.futures_account()
        available = float(data["availableBalance"])
        total = float(data["totalWalletBalance"])
        if total == 0: return
        ratio = available / total
        if ratio < n_stop:
            send_telegram_alert(f"🔴 Margin <10%: {ratio:.2%} — Full liquidation.")
            reduce_portfolio_by(1.0)
        elif ratio < n_sell:
            send_telegram_alert(f"🚨 Margin <20%: {ratio:.2%} — Sell 30% of portfolio.")
            reduce_portfolio_by(0.3)
        elif ratio < n_warn:
            send_telegram_alert(f"⚠️ Margin <30%: {ratio:.2%}. Monitor.")
    except Exception as e:
        send_telegram_alert(f"❌ Margin check failed: {e}")

def reduce_portfolio_by(percent, portfolio_file="top_coins.json"):
    portfolio = load_json_file(portfolio_file)
    recovery_plan = {}
    for pos in portfolio:
        symbol = pos["symbol"]
        qty = pos["quantity"] * percent
        market_type = market_type_map.get(symbol, "SPOT")
        place_order(symbol, "SELL", qty, market_type=market_type)
        recovery_plan[symbol] = {"symbol": symbol, "quantity": qty, "market_type": market_type}
    if recovery_plan:
        save_json_file(recovery_plan, "recovery_plan.json")


def cover_short_positions(current_bottom_roc_symbols):
    """Buy to cover shorts that are no longer in the bottom 10 or when BTC > MA50"""
    short_positions = load_json_file("short_positions.json")
    if not short_positions:
        return
    remaining_shorts = []
    for pos in short_positions:
        symbol = pos["symbol"]
        if symbol not in current_bottom_roc_symbols:
            try:
                qty = pos["quantity"]
                market_type = market_type_map.get(symbol, "FUTURES")
                place_order(symbol, "BUY", qty, market_type=market_type)
                log_transaction("COVER", symbol, qty)
                send_telegram_alert(f"✅ Covered short for {symbol} at market.")
            except Exception as e:
                send_telegram_alert(f"❌ Failed to cover short for {symbol}: {e}")
                remaining_shorts.append(pos)
        else:
            remaining_shorts.append(pos)
    save_json_file(remaining_shorts, "short_positions.json")


def recover_positions(filename="recovery_plan.json"):
    recovery_plan = load_json_file(filename)
    if not recovery_plan: return
    for symbol, entry in recovery_plan.items():
        qty = entry["quantity"]
        mt = entry.get("market_type", "SPOT")
        place_order(symbol, "BUY", qty, market_type=mt)
    save_json_file({}, filename)

def load_json_file(filename):
    if not os.path.exists(filename): return {}
    with open(filename, "r") as f:
        return json.load(f)

def save_json_file(data, filename):
    with open(filename, "w") as f:
        json.dump(data, f, indent=2)

@rate_limit_retry
def get_symbol_info(symbol):
    """Fetch symbol-specific information, including price and quantity filters."""
    exchange_info = client.get_exchange_info()
    for s in exchange_info['symbols']:
        if s['symbol'] == symbol:
            return s
    return None


def validate_order(symbol, price, quantity):
    """
    Validate and adjust price and quantity for an order based on Binance filters.
    Ensures compliance with PRICE_FILTER, LOT_SIZE, and MIN_NOTIONAL.

    Parameters:
        symbol (str): The trading pair symbol (e.g., 'BTCUSDT').
        price (float): The price to validate and adjust.
        quantity (float): The quantity to validate and adjust.
    Returns:
        tuple: Validated and adjusted (price, quantity).
    """
    symbol_info = get_symbol_info(symbol)
    if not symbol_info:
        raise ValueError(f"Symbol information not found for {symbol}")

    # Adjust price using PRICE_FILTER
    for f in symbol_info['filters']:
        if f['filterType'] == 'PRICE_FILTER':
            tick_size = float(f['tickSize'])
            # Round down to nearest tick size
            price = math.floor(price / tick_size) * tick_size

    # Adjust quantity using LOT_SIZE
    for f in symbol_info['filters']:
        if f['filterType'] == 'LOT_SIZE':
            step_size = float(f['stepSize'])
            min_qty = float(f['minQty'])

            # Automatically adjust quantity to minQty if it's too small
            if quantity < min_qty:
                quantity = min_qty

            # Calculate precision based on step size
            precision = int(round(-math.log(step_size, 10), 0))

            # Round down to nearest step size
            quantity = math.floor(quantity / step_size) * step_size

            # Explicitly round to the required precision
            quantity = round(quantity, precision)

            if quantity < min_qty:
                raise ValueError(f"Quantity {quantity} is below the minimum allowed {min_qty}")

    # Validate against MIN_NOTIONAL
    for f in symbol_info['filters']:
        if f['filterType'] == 'MIN_NOTIONAL':
            min_notional = float(f['minNotional'])
            if price * quantity < min_notional:
                # Adjust quantity to meet minNotional if possible
                adjusted_quantity = min_notional / price
                quantity = math.floor(adjusted_quantity / step_size) * step_size
                quantity = round(quantity, precision)
                if quantity < min_qty:
                    raise ValueError(
                        f"Order value {price * quantity} is below the minimum notional value {min_notional}"
                    )

    return price, quantity


def round_quantity(symbol, quantity):
    """Round the quantity to the correct precision based on LOT_SIZE."""
    symbol_info = get_symbol_info(symbol)
    if not symbol_info:
        raise ValueError(f"Symbol information not found for {symbol}")

    for f in symbol_info['filters']:
        if f['filterType'] == 'LOT_SIZE':
            step_size = float(f['stepSize'])
            precision = int(round(-math.log(step_size, 10), 0))
            return round(quantity, precision)
    return quantity


def round_price(symbol, price):
    """Round the price to the correct precision based on PRICE_FILTER."""
    symbol_info = get_symbol_info(symbol)
    if not symbol_info:
        raise ValueError(f"Symbol information not found for {symbol}")

    for f in symbol_info['filters']:
        if f['filterType'] == 'PRICE_FILTER':
            tick_size = float(f['tickSize'])
            precision = int(round(-math.log(tick_size, 10), 0))
            price = round(price, precision)

    return price


@rate_limit_retry
def place_vwap_order(symbol, side, allocation, vwap, all_orders_summary):
    """Place limit order at VWAP and add summary to the provided summary dictionary."""
    try:
        # Fetch VWAP and calculate quantity/price
        price = vwap * 0.98
        quantity = allocation / price

        # Adjust quantity and price
        price, quantity = validate_order(symbol, price, quantity)
        # quantity = round_quantity(symbol, quantity)
        # price = round_price(symbol, price)

        # Place limit order at VWAP
        order = client.create_order(
            symbol=symbol,
            side=side,
            type=Client.ORDER_TYPE_LIMIT,
            timeInForce=Client.TIME_IN_FORCE_GTC,
            quantity=quantity,
            price=f"{price:.8f}"
        )
        # Append order details to the summary
        all_orders_summary[symbol] = {
            "Symbol": symbol,
            "Action": side,
            "Strategy Quantity": quantity,
            "Binance Quantity": order.get("executedQty", 0),
            "Strategy Price": f"{price:.2f}",
            "Filled": order.get("fills")[0].get("price") if order.get("fills") else "-",
            "Cost": float(order.get("executedQty", 0)) * float(price),
            "Order ID": order["orderId"],
            "Live": (get_live_price(symbol) or 0) * quantity,
            "Status": order.get("status", "PENDING")
        }

        # Load and update the pending orders file
        pending_orders = load_json_file(filename='pending_orders.json')
        pending_orders[symbol] = {
            "orderId": order["orderId"],
            "side": side,
            "quantity": quantity,
            "price": price,
        }
        save_json_file(pending_orders, "pending_orders.json")

    except Exception as e:
        # Log error in the summary with the same structure
        all_orders_summary[symbol] = {
            "Symbol": symbol,
            "Action": side,
            "Strategy Quantity": allocation / vwap,
            "Binance Quantity": 0,
            "Strategy Price": f"{vwap:.2f}",
            "Filled": "-",
            "Cost": 0,
            "Order ID": "-",
            "Live": 0,
            "Status": f"ERROR: {str(e)}"
        }


def send_order_summary_notification(order_summary, portfolio_balance=None):
    """Send a user-friendly summary notification for monitored orders."""
    # Get the current time in UTC
    current_time = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    if not order_summary:
        # Graceful handling for an empty summary (just in case)
        send_telegram_alert(
            f" *Order Monitoring Summary* ({current_time}): No actions to report."
        )
        return

    # Helper function for safely formatting numeric values
    def safe_format(value):
        try:
            return f"{float(value):.2f}" if value is not None else "-"
        except (ValueError, TypeError):
            return "-"

    # Initialize the message
    message = f" *Order Monitoring Summary* ({current_time}):\n\n"
    message += (
        "\n"
        f"{'Symbol':<8} {'Action':<10} {'Str Qty':<10} {'Bin Qty':<10} "
        f"{'Price':<8} {'Cost':<8} {'LIVE':<8} {'Status':<8}\n"
        + "-" * 65
    )

    for symbol, details in order_summary.items():
        clean_symbol = symbol.replace("USDT", "")
        if "Error Message" in details:
            # Error case
            message += f"\n{clean_symbol:<8} ERROR: {details['Error Message']}"
        else:
            # Safely extract and format details with fallback for invalid fields
            try:
                action = details.get("Action", "-")
                strategy_qty = safe_format(details.get("Strategy Quantity"))
                binance_qty = safe_format(details.get("Binance Quantity"))
                price = safe_format(details.get("Filled"))
                cost = safe_format(details.get("Cost"))
                live_value = safe_format(details.get("Live"))
                status = details.get("Status", "-")

                # Format the order details
                message += (
                    f"\n"
                    f"{clean_symbol:<8} {action:<12} {strategy_qty:<10} {binance_qty:<10} "
                    f"{price:<8} {cost:<8} {live_value:<8} {status:<8}\n"
                )
            except Exception as e:
                # Handle unexpected errors gracefully
                message += f"\n{clean_symbol:<8} ERROR: {str(e)}"

    # Add portfolio balance if provided
    if portfolio_balance is not None:
        message += f"\n💼 *Order Monitoring Complete*:\nPortfolio Balance: ${portfolio_balance:.2f}"

    # Add the portfolio value at the end of the message
    try:
        portfolio_live_value = get_portfolio_value()
        message += "\n\nPortfolio LIVE: $" + \
            f"{float(portfolio_live_value):.2f}"
    except (ValueError, TypeError):
        message += "\n\nPortfolio LIVE: -"

    # Send the formatted message
    send_telegram_alert(message)


def send_batch_telegram_alert(all_orders_summary, portfolio_balance=None):
    """Send a user-friendly Telegram alert summarizing all orders."""
    # Get the current time in UTC
    current_time = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    # Start the message with a header
    message = f"📊 *Rebalance Orders Summary* ({current_time}):\n\n"
    message += (
        "\n"
        f"{'Symbol':<8} {'Action':<8} {'Str Qty':<10} {'Bin Qty':<10} "
        f"{'Price':<8} {'Cost':<8} {'LIVE':<8} {'Status':<8}\n"
        + "-" * 65
    )

    for symbol, details in all_orders_summary.items():
        clean_symbol = symbol.replace("USDT", "")
        if "Error" in details:
            # If there's an error, show it in the message
            message += f"{symbol:<10} ERROR: {details['Error']}\n"
        else:
            # Safely extract and format details with fallback for missing or invalid fields
            def safe_format(value):
                try:
                    return f"{float(value):.2f}" if value is not None else "-"
                except (ValueError, TypeError):
                    return "-"

            action = details.get("Action", "-")
            strategy_qty = safe_format(details.get("Strategy Quantity"))
            binance_qty = safe_format(details.get("Binance Quantity"))
            price = safe_format(details.get("Filled"))
            cost = safe_format(details.get("Cost"))
            live_value = safe_format(details.get("Live"))
            # order_id = details.get("Order ID", "-")
            status = details.get("Status", "-")

            # Format the order details
            message += (
                f"\n"
                f"{clean_symbol:<10} {action:<8} {strategy_qty:<8} {binance_qty:<8} "
                f"{price:<8} {cost:<8} {live_value:<8} {status:<8}\n"
            )

    # Add portfolio balance if provided
    if portfolio_balance is not None:
        message += f"\n💼 *Rebalance Complete*:\nPortfolio Balance: ${portfolio_balance:.2f}"

    # Add the portfolio value at the end of the message
    try:
        portfolio_live_value = get_portfolio_value()
        message += "\n\nPortfolio LIVE: $" + f"{float(portfolio_live_value):.2f}"
    except (ValueError, TypeError):
        message += "\n\nPortfolio LIVE: -"

    # End message with closing format
    message += "\n"

    # Send the formatted message
    send_telegram_alert(message)




def record_daily_portfolio_value():
    portfolio_value = get_portfolio_value()  # Call the function to get the value
    if portfolio_value is None:
        print("Failed to record portfolio value.")
        return

    # File to store daily values
    file_path = "portfolio_values.json"

    # Load existing data if the file exists
    if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
        with open(file_path, "r") as file:
            try:
                data = json.load(file)
            except json.JSONDecodeError as e:
                print(f"Error reading JSON: {e}")
                data = []  # Initialize with an empty list if the file is corrupted
    else:
        data = []

    # Add today's portfolio value only if today's date is not already in the file
    today = datetime.now().strftime("%Y-%m-%d")
    if any(entry["date"] == today for entry in data):
        print(f"Portfolio value for {today} is already recorded. Skipping.")
        return

    # Add today's portfolio value
    data.append({"date": today, "value": portfolio_value})

    # Save updated data
    with open(file_path, "w") as file:
        json.dump(data, file, indent=4)

    print(f"Portfolio value for {today} recorded: ${portfolio_value}")

def generate_cumulative_graph(image_path="portfolio_graph.png"):
    file_path = "portfolio_values.json"

    if not os.path.exists(file_path):
        print("No data available to plot.")
        return

    # Load data
    with open(file_path, "r") as file:
        try:
            data = json.load(file)
        except json.JSONDecodeError:
            print("Invalid or corrupted data file. No graph will be generated.")
            return

    if not data:
        print("Data file is empty. No graph will be generated.")
        return

    # Extract dates and values
    dates = [datetime.strptime(item["date"], "%Y-%m-%d") for item in data]
    values = [item["value"] for item in data]

    if not dates or not values:
        print("No valid data to plot.")
        return

    # Plot the data
    plt.figure(figsize=(12, 6))
    plt.plot(dates, values, marker="o", linestyle="-")
    plt.title("Day Start Portfolio Value Over Time")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value (USDT)")

    # Format x-axis ticks
    plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    plt.gca().xaxis.set_minor_locator(mdates.MonthLocator())
    plt.gca().xaxis.set_minor_formatter(mdates.DateFormatter("%b"))

    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()

    # Save the graph as an image
    plt.savefig(image_path)
    plt.close()
    print(f"Graph saved as {image_path}")
    send_telegram_alert(image_path=image_path)


def monitor_orders(filename="pending_orders.json"):
    """Check all open orders at 12 PM UTC and convert unfilled orders to market orders."""
    try:
        if is_paused:
            send_telegram_alert(
                "🚫 Order monitoring skipped because tasks are paused.")
            return

        # Load pending orders from file
        pending_orders = load_json_file(filename)
        order_summary = {}

        for symbol, order in list(pending_orders.items()):
            try:
                # Fetch the order status from Binance
                order_status = client.get_order(
                    symbol=symbol, orderId=order['orderId']
                )
                # Safely convert executedQty to float
                qty = float(order_status.get('executedQty', 0))

                # Handle unfilled or partially filled orders
                if order_status['status'] in ['NEW', 'PARTIALLY_FILLED']:
                    # Cancel the unfilled or partially filled order
                    client.cancel_order(
                        symbol=symbol, orderId=order['orderId'])
                    remaining_quantity = max(
                        0, float(order_status.get('origQty', 0)) - qty
                    )  # Ensure remaining quantity is non-negative

                    # Place a market order for the remaining quantity
                    if remaining_quantity > 0:
                        price = get_live_price(symbol)
                        market_order = client.create_order(
                            symbol=symbol,
                            side=order['side'],
                            type=Client.ORDER_TYPE_MARKET,
                            quantity=(validate_order(symbol, price, remaining_quantity))[1]
                        )
                        # Extract market order cost and price
                        market_fills = market_order.get('fills', [])
                        market_price = float(
                            market_fills[0]['price']
                        ) if market_fills else 0
                        market_cost = float(
                            remaining_quantity * market_price
                        ) if market_price else 0

                        order_summary[symbol] = {
                            "Symbol": symbol,
                            "Action": "Market Order Placed",
                            "Strategy Quantity": order_status.get('origQty', "-"),
                            "Binance Quantity": remaining_quantity,
                            "Filled": "Market",
                            "Cost": market_cost,
                            "Order ID": market_order.get('orderId', "-"),
                            "Live": (get_live_price(symbol) or 0) * qty,
                            "Status": market_order.get('status', 'FILLED')
                        }
                    else:
                        order_summary[symbol] = {
                            "Symbol": symbol,
                            "Action": "No Remaining Quantity",
                            "Strategy Quantity": order_status.get('origQty', "-"),
                            "Binance Quantity": qty,
                            "Filled": "-",
                            "Cost": 0,
                            "Order ID": order['orderId'],
                            "Live": (get_live_price(symbol) or 0) * qty,
                            "Status": "Skipped"
                        }

                    # Remove the order from the tracking file
                    del pending_orders[symbol]

                elif order_status['status'] == 'FILLED':
                    # If the order is already filled, no further action is required
                    filled_price = float(order_status.get('price', 0))
                    order_summary[symbol] = {
                        "Symbol": symbol,
                        "Action": "Already Filled",
                        "Strategy Quantity": order_status.get('origQty', "-"),
                        "Binance Quantity": qty,
                        "Filled": "-",
                        "Cost": qty * filled_price if filled_price else 0,
                        "Order ID": order['orderId'],
                        "Live": (get_live_price(symbol) or 0) * qty,
                        "Status": "FILLED"
                    }
                    del pending_orders[symbol]

                else:
                    order_summary[symbol] = {
                        "Symbol": symbol,
                        "Action": "No Action Taken",
                        "Strategy Quantity": order_status.get('origQty', "-"),
                        "Binance Quantity": qty,
                        "Filled": "-",
                        "Cost": 0,
                        "Order ID": order['orderId'],
                        "Live": (get_live_price(symbol) or 0) * qty,
                        "Status": order_status['status']
                    }

            except Exception as e:
                # Catch individual order errors and record them
                order_summary[symbol] = {
                    "Symbol": symbol,
                    "Action": "Error",
                    "Strategy Quantity": "-",
                    "Binance Quantity": "-",
                    "Filled": "-",
                    "Cost": 0,
                    "Order ID": order.get('orderId', "-"),
                    "Live": 0,
                    "Status": "ERROR",
                    "Error Message": str(e)
                }

        # Save the updated pending orders file
        save_json_file(pending_orders, "pending_orders.json")
        if not order_summary:
            send_telegram_alert(
                " Order Monitoring Summary: No pending orders in the file. No actions required."
            )
            return

        send_order_summary_notification(order_summary)

    except Exception as e:
        send_telegram_alert(f"❌ Error monitoring orders: {e}")




def handle_telegram_commands():
    global is_paused
    offset = None  # Tracks the last processed update

    while True:
        try:
            # Fetch updates from Telegram
            url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/getUpdates"
            params = {"timeout": 10, "offset": offset}
            response = requests.get(url, params=params)

            # Handle successful responses
            if response.status_code == 200:
                updates = response.json().get("result", [])
                for update in updates:
                    offset = update["update_id"] + 1
                    if "message" in update and "text" in update["message"]:
                        command = update["message"]["text"].strip()
                        print_log(f"Received command: {command}")

                        # Process commands
                        if command == "/start":
                            send_telegram_alert(
                                "Rebalance Portfolio initiated and all scheduled tasks resumed (if previously stopped). ✅"
                            )
                            is_paused = False  # Resume the tasks
                            rebalance_portfolio()
                        elif command == "/stop":
                            send_telegram_alert(
                                "Manual override: Selling all positions and pausing all scheduled tasks. 🚨"
                            )
                            portfolio = load_json_file(filename="top_coins.json")
                            sell_all_positions(portfolio)
                            is_paused = True  # Pause the tasks

                        elif command == "/restart":
                            send_telegram_alert("Manual override: Resuming all scheduled tasks. 🔄")
                            is_paused = False  # Resume the tasks
                        elif command == "/graph":
                            daily_portfolio_task()
                        else:
                            send_telegram_alert(f"Unknown command received: {command} ❓")

            else:
                print_log(f"Error fetching Telegram updates: {response.text}")

        except requests.exceptions.RequestException as e:
            print_log(f"Error in Telegram communication: {e}")

        time.sleep(1)  # Avoid spamming Telegram API


def get_live_price(symbol):
    try:
        # Fetch ticker price for the given symbol
        ticker = client.get_symbol_ticker(symbol=symbol)
        return float(ticker["price"])
    except Exception as e:
        print(f"Error fetching live price for {symbol}: {e}")
        return None


def log_transaction(action, symbol, quantity, roc30=None, vwap=None, filename="coin_transactions.csv"):
    try:
        """
        Log transaction details to a CSV file.

        Args:
            action (str): 'BUY' or 'SELL'.
            symbol (str): The coin symbol.
            quantity (float): Quantity of the coin.
            roc30 (float, optional): Rate of Change over 30 days. Defaults to None.
            vwap (float, optional): Volume Weighted Average Price. Defaults to None.
            filename (str): Name of the CSV file. Defaults to 'coin_transactions.csv'.
        """
        current_time_utc = datetime.now(
            timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        data = {
            "time": [current_time_utc],
            "action": [action],
            "symbol": [symbol],
            "quantity": [quantity],
            "ROC30": [roc30],
            "VWAP": [vwap]
        }
        df = pd.DataFrame(data)

        # Check if the file exists to determine if the header should be written
        file_exists = os.path.isfile(filename)

        # Append to CSV; write header only if the file does not exist
        df.to_csv(filename, mode='a', index=False, header=not file_exists)
    except Exception as e:
        print_log(f"log_transaction {e}")


def sell_all_positions(portfolio, all_orders_summary={}):
    current_time_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    if portfolio:
        send_telegram_alert(
            f"[{current_time_utc}] BTC < 50MA → Selling all positions. 📉")

        updated_portfolio = portfolio.copy()  # Create a copy of the portfolio
        for pos in portfolio:
            symbol = pos["symbol"]
            quantity = pos["quantity"]
            # Attempt to place a Market SELL order
            try:
                price = get_live_price(symbol)
                sell_order = client.create_order(
                    symbol=symbol,
                    side="SELL",
                    type=Client.ORDER_TYPE_MARKET,
                    quantity=(validate_order(symbol, price, quantity))[1]
                )
                all_orders_summary[symbol] = {
                    "Symbol": symbol,
                    "Action": "SELL",
                    "Strategy Quantity": quantity,
                    "Binance Quantity": sell_order.get('executedQty', 0),
                    "Strategy Price": "Market",
                    "Filled": sell_order.get('fills')[0].get('price') if sell_order.get('fills') else "-",
                    "Cost": float(sell_order.get('executedQty', 0)) * float(
                        sell_order.get('fills')[0].get('price')) if sell_order.get('fills') else 0,
                    "Order ID": sell_order["orderId"],
                    "Live": (get_live_price(symbol) or 0) * quantity,
                    "Status": sell_order.get('status', '-')
                }
                log_transaction("SELL", symbol, quantity)
                # Remove the successfully processed position from the portfolio
                updated_portfolio.remove(pos)

            except Exception as e:
                all_orders_summary[symbol] = {
                    "Side": "SELL",
                    "Error": str(e)
                }
                send_telegram_alert(
                    f"Failed to place SELL order for {symbol}: {str(e)} ❌")

        # Save the updated portfolio (removing successful sales only)
        save_json_file(updated_portfolio)
        save_json_file({}, filename='pending_orders.json')

        # Send consolidated notification after processing all sells
        send_batch_telegram_alert(all_orders_summary)
        send_telegram_alert("BTC < 50MA → All positions processed and portfolio updated. ✅")

    else:
        send_telegram_alert(f"[{current_time_utc}] BTC < 50MA → No positions found to sell. No action needed. 🔍")

    return


def daily_portfolio_task():
    record_daily_portfolio_value()
    generate_cumulative_graph()

def rebalance_portfolio():
    """Rebalance the portfolio at 0000 UTC, based on BTC 50MA condition."""

    if is_paused:
        send_telegram_alert(f"Rebalancing skipped because tasks are paused. ⏸️")
        return

    current_time_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    all_orders_summary = {}

    # Fetch the current account balance dynamically
    try:
        spot_balance_usdt = 0
        future_balance_usdt = 0

        try:
            # Spot balance
            account_info_spot = client.get_account()
            spot_balance_usdt = sum(
                float(balance['free']) + float(balance['locked'])
                for balance in account_info_spot['balances']
                if balance['asset'] == 'USDT'
            )
        except Exception as e:
            print_log(f"Error fetching SPOT balance: {e}")

        try:
            # Futures balance
            account_info_futures = client_future.futures_account()
            future_balance_usdt = float(account_info_futures.get("totalWalletBalance", 0))
        except Exception as e:
            print_log(f"Error fetching FUTURES balance: {e}")

        total_balance_usdt = spot_balance_usdt + future_balance_usdt
        allocation_percentage = 0.10  # Allocate 10% of the total balance
        allocated_capital = total_balance_usdt * allocation_percentage

        if allocated_capital <= 0:
            send_telegram_alert("Insufficient balance for rebalancing.")
            return

    except Exception as e:
        print_log(f"Error fetching account balance: {e}")
        send_telegram_alert(f"Error fetching account balance: {e}")
        return

    # 1) Load the portfolio (the coins we previously bought under this strategy)
    portfolio = load_json_file(filename="top_coins.json")

    # 2) Check if BTC is above 50MA or not
    if not is_strategy_active('BTCUSDT'):
        # -----------------------------------------------------------------------
        # BTC BELOW 50MA → Market-Sell All from the file
        # -----------------------------------------------------------------------
        sell_all_positions(portfolio)

    # -----------------------------------------------------------------------
    # BTC BELOW 50MA → SHORT the Bottom 10 ROC coins
    # -----------------------------------------------------------------------
    send_telegram_alert(
        f"[{current_time_utc}] BTC < 50MA → Enter Short Strategy (Bottom 10 ROC) 🔽")

    bottom_roc_coins = get_bottom_10_coins_usdt()
    if not bottom_roc_coins:
        send_telegram_alert("No coins available for shorting.")
        return

    short_data = {}
    allocation_per_coin = allocated_capital / len(bottom_roc_coins)
    for coin in bottom_roc_coins:
        symbol = coin['symbol']
        try:
            price_data = fetch_ohlcv(symbol)
            if price_data is None or price_data.empty:
                raise ValueError(f"No price data for {symbol}")
            _, _, cross_below = get_ma20_signals(price_data)
            if not cross_below.iloc[-2]:
                raise ValueError(f"{symbol} did not cross below MA20, skipping.")

            vwap = calculate_vwap(price_data)
            if vwap is None:
                raise ValueError(f"Invalid VWAP for {symbol}")

            quantity = allocation_per_coin / vwap

            if USE_DYNAMIC_VOL_SIZING:
                returns = price_data['close'].pct_change()
                try:
                    position_scaler = calculate_volatility_position_size(returns)
                    quantity *= position_scaler
                except:
                    print_log(f"Volatility sizing failed for {symbol}, using fixed quantity.")

            quantity = (validate_order(symbol, vwap * 1.02, quantity))[1]

            short_data[symbol] = {
                "vwap": vwap,
                "quantity": quantity,
                "roc30": coin["ROC30"],
            }

            place_vwap_order(
                symbol=symbol,
                side="SELL",
                allocation=allocation_per_coin,
                vwap=vwap,
                all_orders_summary=all_orders_summary
            )

            log_transaction("SHORT", symbol, quantity, roc30=coin["ROC30"], vwap=vwap)
        except Exception as e:
            send_telegram_alert(f"❌ Error shorting {symbol}: {e}")


    
            short_data[symbol] = {
                "vwap": vwap,
                "quantity": quantity,
                "roc30": coin["ROC30"],
                "time": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
            }

    # Save short positions to file
    save_json_file(
        [{"symbol": s, **d} for s, d in short_data.items()],
        "short_positions.json"
    )

    # -----------------------------------------------------------------------
    # BTC ABOVE 50MA → Normal Rebalance
    # -----------------------------------------------------------------------
    send_telegram_alert(
        f"[{current_time_utc}] BTC > 50MA → Rebalance into Top 10 🔼")

    cover_short_positions([c["symbol"] for c in bottom_roc_coins])

    # 3) Fetch today's top 10 coins
    today_top_coins = get_top_10_coins_usdt()
    if not today_top_coins:
        send_telegram_alert("No coins available for rebalancing.")
        return

    new_symbols = [c["symbol"] for c in today_top_coins]
    portfolio_dict = {p["symbol"]: p for p in portfolio}

    # 4) Market-sell any coin that's in the file but not in today's top 10
    coins_to_sell = [symbol for symbol in portfolio_dict if symbol not in new_symbols]
    coins_to_hold = [symbol for symbol in portfolio_dict if symbol in new_symbols]

    if coins_to_sell:
        pending_orders = load_json_file(filename="pending_orders.json")
        sell_message = f"Selling positions not in today's Top 10 ({current_time_utc}):\n"
        for symbol in coins_to_sell:
            qty = portfolio_dict[symbol]["quantity"]
            retry_attempts = 3
            delay = 1  # Initial delay for exponential backoff

            while retry_attempts > 0:
                try:
                    price = get_live_price(symbol)
                    sell_order = client.create_order(
                        symbol=symbol,
                        side="SELL",
                        type=Client.ORDER_TYPE_MARKET,
                        quantity=(validate_order(symbol, price, qty))[1]
                    )
                    all_orders_summary[symbol] = {
                        "Symbol": symbol,
                        "Action": "SELL",
                        "Strategy Quantity": qty,
                        "Binance Quantity": sell_order.get('executedQty', 0),
                        "Strategy Price": "Market",
                        "Filled": sell_order.get('fills')[0].get('price') if sell_order.get('fills') else "-",
                        "Cost": float(sell_order.get('executedQty', 0)) * float(
                            sell_order.get('fills')[0].get('price')) if sell_order.get('fills') else 0,
                        "Order ID": sell_order["orderId"],
                        "Live": (get_live_price(symbol) or 0) * qty,
                        "Status": sell_order.get('status', '-')
                    }
                    del portfolio_dict[symbol]

                    # Safely remove the symbol from pending_orders
                    if symbol in pending_orders:
                        del pending_orders[symbol]

                    log_transaction("SELL", symbol, qty)
                    break
                except Exception as e:
                    retry_attempts -= 1
                    if retry_attempts == 0:
                        all_orders_summary[symbol] = {
                            "Side": "SELL",
                            "Error": str(e)
                        }
                        send_telegram_alert(f"Failed to sell {symbol} after retries: {e}")
                    else:
                        time.sleep(delay)
                        delay *= 2  # Exponential backoff
        save_json_file(pending_orders, "pending_orders.json")
        send_telegram_alert(sell_message)

    # 5) Market-buy  any coins in today's Top 10 but not in portfolio
    coins_to_buy = [symbol for symbol in new_symbols if symbol not in portfolio_dict]
    if coins_to_buy:
        buy_message = f"Buying new coins in today's Top 10 (as of {current_time_utc}):\n"
        allocation_per_coin = allocated_capital / len(today_top_coins)
        coin_map = {c["symbol"]: c for c in today_top_coins}
        # Temporary dictionary to store calculated values
        buy_data = {}

        # First loop: Calculate and generate the buy message
        for symbol in coins_to_buy:
            try:
                coin_data = coin_map[symbol]

                # --- MA20 Entry Condition ---
                price_data = fetch_ohlcv(symbol)
                if price_data is None or price_data.empty:
                    raise ValueError(f"No price data for {symbol}")
                _, cross_above, _ = get_ma20_signals(price_data)
                if not cross_above.iloc[-2]:
                    raise ValueError(f"{symbol} did not cross above MA20, skipping.")

                # --- VWAP Calculation ---
                vwap = calculate_vwap(price_data)
                if vwap is None:
                    raise ValueError(f"Invalid VWAP for {symbol}")

                quantity = allocation_per_coin / vwap

                # --- Volatility Sizing ---
                if USE_DYNAMIC_VOL_SIZING:
                    returns = price_data['close'].pct_change()
                    try:
                        position_scaler = calculate_volatility_position_size(returns)
                        quantity *= position_scaler
                    except:
                        print_log(f"Volatility sizing failed for {symbol}, using fixed quantity.")

                quantity = (validate_order(symbol, vwap * 0.98, quantity))[1]

                # Store data in temporary dictionary for reuse
                buy_data[symbol] = {
                    "vwap": vwap,
                    "quantity": quantity,
                    "roc30": coin_data["ROC30"],
                }

                buy_message += (
                    f"- {symbol}: ROC30={coin_data['ROC30']:.3f}%, VWAP={vwap:.4f}, "
                    f"Allocation={allocation_per_coin:.2f}, Qty={quantity}\n"
                )

                log_transaction("BUY", symbol, quantity,
                                roc30=coin_data["ROC30"], vwap=vwap)
            except Exception as e:
                buy_message += f"- Error buying {symbol}: {e}\n"
        # Send the message before placing any orders
        send_telegram_alert(buy_message)

        # Second loop: Place the BUY orders and update the portfolio
        for symbol, data in buy_data.items():
            try:
                # Reuse calculated values
                place_vwap_order(
                    symbol=symbol,
                    side="BUY",
                    allocation=allocation_per_coin,
                    vwap=data["vwap"],
                    all_orders_summary=all_orders_summary
                )

                # Check if the order was successfully added to the summary
                if symbol in all_orders_summary and "Error" not in all_orders_summary[symbol]:
                    # Update the portfolio dictionary only if the order was successful
                    portfolio_dict[symbol] = {
                        "symbol": symbol,
                        "quantity": data["quantity"],
                        "time": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
                    }
                else:
                    raise Exception(f"Order placement for {symbol} failed. Error: {all_orders_summary.get(symbol)}")
            except Exception as e:
                all_orders_summary[symbol] = {
                    "Side": "BUY",
                    "Error": str(e)
                }
    else:
        send_telegram_alert(f"✅ Already holding today's Top 10 coins. No buys are needed.")
        return

    # 6) Save the updated portfolio (convert dict back to list)
    updated_portfolio = list(portfolio_dict.values())
    save_json_file(updated_portfolio, "top_coins.json")

        # Generate a summary of held coins for the Telegram notification
    hold_message = ''
    if coins_to_hold:
        hold_message += f"Holding positions (as of {current_time_utc}):\n\n\n"
        for symbol in coins_to_hold:
            hold_message += (
                f"- {symbol}: Qty={portfolio_dict[symbol]['quantity']}, "
                f"Last Updated={portfolio_dict[symbol]['time']}\n"
            )

    # Combine all notifications into a single summary
    summary_message = hold_message + "\nRebalance complete. Portfolio successfully updated. ✅"
    if all_orders_summary:
        send_batch_telegram_alert(all_orders_summary)

    send_telegram_alert(summary_message)


if __name__ == "__main__":
    # rebalance_portfolio()
    threading.Thread(target=handle_telegram_commands, daemon=True).start()
    schedule.every(5).minutes.do(check_margin_health)
    schedule.every().day.at("00:00").do(daily_portfolio_task)
    schedule.every().day.at("00:01").do(rebalance_portfolio)
    schedule.every().day.at("00:05").do(recover_positions)
    schedule.every().day.at("12:00").do(monitor_orders)

    while True:
        if not is_paused:  # Only run tasks if not paused
            schedule.run_pending()
        time.sleep(60)  # Check every minute
