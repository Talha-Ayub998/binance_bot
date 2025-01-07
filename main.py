import pandas as pd
import requests
import time
import threading
from binance.client import Client
from datetime import datetime, timedelta, timezone
import schedule
import math
import json
import os
from dotenv import load_dotenv

# Load the .env file
load_dotenv()

# Retrieve the environment variables
API_KEY = os.getenv("API_KEY")
API_SECRET = os.getenv("API_SECRET")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# Initialize Binance Client
client = Client(API_KEY, API_SECRET, testnet=True)

# Globals
PORTFOLIO_VALUE = 100  # Example portfolio value
TOP_N_COINS = 10
# Global dictionary to track orders
orders = {}


def send_telegram_alert(message):
    """Send a Telegram alert."""
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message}
    try:
        requests.post(url, json=payload, timeout=10)
    except Exception as e:
        print(f"Error sending Telegram alert: {e}")


def fetch_ohlcv(symbol, interval="1h", lookback="1 day ago UTC"):
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
            print(f"Error fetching data from Binance: {response.text}")
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
        print(f"Error in fetch_ohlcv: {e}")
        return pd.DataFrame()


def calculate_roc30(data):
    """
    Calculate the 30-day Rate of Change (ROC) for the latest available day.
    """
    latest_close = data['close'].iloc[-1]
    old_close = data['close'].shift(30).iloc[-1]
    if pd.isna(old_close):
        return None
    return ((latest_close - old_close) / old_close) * 100


def calculate_vwap(data):
    """Calculate VWAP."""
    return (data['close'] * data['volume']).sum() / data['volume'].sum()


def is_strategy_active(symbol, ma_period=50, interval="1d", lookback="51 days ago UTC"):
    """Check if the symbol's closing price(up to yesterday) is above its moving average(MA)."""
    try:
        data = fetch_ohlcv(symbol, interval=interval, lookback=lookback)
        if data.empty or len(data) < ma_period + 1:
            print(
                f"Not enough data to calculate {ma_period}-day MA for {symbol}.")
            return False
        data = data.iloc[:-1]
        ma = data['close'].rolling(window=ma_period).mean().iloc[-1]
        return data['close'].iloc[-1] > ma
    except Exception as e:
        print(f"Error in is_strategy_active: {e}")
        return False


# ---- Strategy Functions ----
def get_top_10_coins_usdt():
    """Get symbols of the top 10 USDT pairs by ROC30 and volume."""
    symbols = [s['symbol'] for s in client.get_exchange_info(
    )['symbols'] if s['quoteAsset'] == 'USDT']
    filtered_coins = []
    for symbol in symbols:
        try:
            data = fetch_ohlcv(symbol, interval="1d",
                               lookback="31 days ago UTC")
            # Skip if data has less than 30 rows
            if len(data) < 30:
                print(
                    f"{symbol}: Insufficient data (less than 30 days). Skipping...")
                continue
            roc30 = calculate_roc30(data)
            volume = data['volume'][-7:].sum()
            if volume and volume > 10_000_000:
                filtered_coins.append({'symbol': symbol, 'ROC30': roc30})
            else:
                print(f"{symbol}: Insufficient liquidity (volume = {volume}). Skipping...")
        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")

    # Sort by ROC30 and pick the top N symbols
    top_symbols = sorted(
        filtered_coins, key=lambda x: x['ROC30'], reverse=True)[:TOP_N_COINS]
    # top_symbols = [coin['symbol'] for coin in sorted(
    #     filtered_coins, key=lambda x: x['ROC30'], reverse=True)[:TOP_N_COINS]]

    return top_symbols


def get_symbol_info(symbol):
    """Fetch symbol-specific information, including price and quantity filters."""
    exchange_info = client.get_exchange_info()
    for s in exchange_info['symbols']:
        if s['symbol'] == symbol:
            return s
    return None


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
            return round(price, precision)
    return price


def place_vwap_order(symbol, side, allocation):
    """Place limit order at VWAP and save to pending orders file."""
    try:
        # Fetch VWAP and calculate quantity/price
        data = fetch_ohlcv(symbol)
        vwap = calculate_vwap(data)
        price = vwap if side == "BUY" else vwap * 1.02
        quantity = allocation / price

        # Adjust quantity and price
        quantity = round_quantity(symbol, quantity)
        price = round_price(symbol, price)

        # Place limit order at VWAP
        order = client.create_order(
            symbol=symbol,
            side=side,
            type=Client.ORDER_TYPE_LIMIT,
            timeInForce=Client.TIME_IN_FORCE_GTC,
            quantity=quantity,
            price=f"{price:.8f}"
        )
        send_telegram_alert(f"{side} order placed at VWAP for {symbol}: Quantity={quantity}, Price={price:.2f}, Order Details={order}")

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
        send_telegram_alert(f"Error placing VWAP order for {symbol}: {e}")


def monitor_orders(filename="pending_orders.json"):
    """Check all open orders at 12 PM UTC and convert unfilled orders to market orders."""
    try:
        # Load pending orders from file
        pending_orders = load_json_file(filename)

        for symbol, order in list(pending_orders.items()):
            try:
                # Fetch the order status from Binance
                order_status = client.get_order(
                    symbol=symbol, orderId=order['orderId']
                )

                # Handle unfilled or partially filled orders
                if order_status['status'] in ['NEW', 'PARTIALLY_FILLED']:
                    # Cancel the unfilled or partially filled order
                    client.cancel_order(
                        symbol=symbol, orderId=order['orderId'])
                    remaining_quantity = float(
                        order_status['origQty']) - float(order_status['executedQty'])

                    # Place a market order for the remaining quantity
                    if remaining_quantity > 0:
                        market_order = client.create_order(
                            symbol=symbol,
                            side=order['side'],
                            type=Client.ORDER_TYPE_MARKET,
                            quantity=round_quantity(symbol, remaining_quantity)
                        )
                        send_telegram_alert(f"{order['side']} market order placed for {symbol}: {market_order}")

                    # Remove the order from the tracking file
                    del pending_orders[symbol]

                elif order_status['status'] == 'FILLED':
                    # If the order is already filled, no further action is required
                    send_telegram_alert(f"Order {order['orderId']} for {symbol} is already filled.")
                    del pending_orders[symbol]

                else:
                    send_telegram_alert(f"Order {order['orderId']} for {symbol} is in status: {order_status['status']}. No action taken.")

            except Exception as e:
                send_telegram_alert(f"Error checking order for {symbol}: {e}")

        # Save the updated pending orders file
        save_json_file(pending_orders, "pending_orders.json")

    except Exception as e:
        send_telegram_alert(f"Error monitoring orders: {e}")


def load_json_file(filename="top_coins.json"):
    """Load a JSON file and return its content. Return a default dictionary or list if the file doesn't exist."""
    if not os.path.exists(filename):
        # Default to an empty dictionary for pending orders
        return {} if filename == "pending_orders.json" else []
    with open(filename, "r") as f:
        data = json.load(f)
        # Ensure pending_orders.json is a dictionary
        if filename == "pending_orders.json" and not isinstance(data, dict):
            return {}
        return data


def save_json_file(portfolio, filename="top_coins.json"):
    with open(filename, "w") as f:
        json.dump(portfolio, f, indent=2)


def rebalance_portfolio():
    """Rebalance the portfolio at 0000 UTC, based on BTC 50MA condition."""
    current_time_utc = datetime.now(
        timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    # 1) Load the portfolio (the coins we previously bought under this strategy)
    portfolio = load_json_file(filename="top_coins.json")

    # 2) Check if BTC is above 50MA or not
    if not is_strategy_active('BTCUSDT'):
        # -----------------------------------------------------------------------
        # BTC BELOW 50MA → Market-Sell All from the file
        # -----------------------------------------------------------------------
        if portfolio:
            send_telegram_alert(f"[{current_time_utc}] BTC < 50MA → Selling all positions in the file.")
            for pos in portfolio:
                symbol = pos["symbol"]
                quantity = pos["quantity"]
                # Market SELL the entire quantity
                try:
                    sell_order = client.create_order(
                        symbol=symbol,
                        side="SELL",
                        type=Client.ORDER_TYPE_MARKET,
                        quantity=round_quantity(symbol, quantity)
                    )
                    send_telegram_alert(f"Market SELL for {symbol} x {quantity}. Order={sell_order}")
                except Exception as e:
                    send_telegram_alert(f"Error selling {symbol}: {e}")

            # Clear the file
            save_json_file([])
            send_telegram_alert("All positions sold. Portfolio file cleared.")
        else:
            send_telegram_alert(f"[{current_time_utc}] BTC < 50MA, but portfolio file is empty. No action needed.")
        return

    # -----------------------------------------------------------------------
    # BTC ABOVE 50MA → Normal Rebalance
    # -----------------------------------------------------------------------
    send_telegram_alert(f"[{current_time_utc}] BTC > 50MA → Rebalance into Top 10")

    # 3) Fetch today's top 10 coins
    today_top_coins = get_top_10_coins_usdt()
    new_symbols = [c["symbol"] for c in today_top_coins]
    portfolio_dict = {p["symbol"]: p for p in portfolio}

    # 4) Market-sell any coin that's in the file but not in today's top 10
    coins_to_sell = [
        symbol for symbol in portfolio_dict if symbol not in new_symbols]
    if coins_to_sell:
        sell_message = f"Selling positions not in today's Top 10 ({current_time_utc}):\n"
        for symbol in coins_to_sell:
            qty = portfolio_dict[symbol]["quantity"]
            try:
                sell_order = client.create_order(
                    symbol=symbol,
                    side="SELL",
                    type=Client.ORDER_TYPE_MARKET,
                    quantity=round_quantity(symbol, qty)
                )
                sell_message += f"- SOLD {symbol} x {qty}\n"
                del portfolio_dict[symbol]
            except Exception as e:
                sell_message += f"- Error selling {symbol}: {e}\n"
        send_telegram_alert(sell_message)

    # 5) Market-buy any new coins that are in today's top 10 but not in the file
    coins_to_buy = [
        symbol for symbol in new_symbols if symbol not in portfolio_dict]
    if coins_to_buy:
        buy_message = f"Buying new coins in today's Top 10 (as of {current_time_utc}):\n"
        allocation_per_coin = PORTFOLIO_VALUE / len(today_top_coins)
        coin_map = {c["symbol"]: c for c in today_top_coins}

        # First loop: Generate the buy message
        for symbol in coins_to_buy:
            coin_data = coin_map[symbol]
            vwap = calculate_vwap(fetch_ohlcv(symbol))
            quantity = allocation_per_coin / vwap
            quantity = round_quantity(symbol, quantity)

            buy_message += (
                f"- {symbol}: ROC30={coin_data['ROC30']:.3f}%, VWAP={vwap:.4f}, "
                f"Allocation={allocation_per_coin:.2f}, Qty={quantity}\n"
            )

        # Send the message before placing any orders
        send_telegram_alert(buy_message)

        # Second loop: Place the BUY orders and update the portfolio
        for symbol in coins_to_buy:
            coin_data = coin_map[symbol]
            place_vwap_order(symbol=symbol, side="BUY",
                             allocation=allocation_per_coin)

            # Update the portfolio dictionary
            portfolio_dict[symbol] = {
                "symbol": symbol,
                "quantity": allocation_per_coin / calculate_vwap(fetch_ohlcv(symbol)),
                "time": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
            }
    else:
        send_telegram_alert("Already holding today's Top 10. No buys needed.")

    # 6) Save the updated portfolio (convert dict back to list)
    updated_portfolio = list(portfolio_dict.values())
    save_json_file(updated_portfolio, "top_coins.json")
    send_telegram_alert("Rebalance complete. Portfolio file updated.")


if __name__ == "__main__":
    # rebalance_portfolio()
    schedule.every().day.at("00:00").do(rebalance_portfolio)
    schedule.every().day.at("12:00").do(monitor_orders)

    while True:
        schedule.run_pending()
        time.sleep(60)
