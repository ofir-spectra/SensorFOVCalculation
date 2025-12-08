import sqlite3
from datetime import datetime, timedelta

DB_FILENAME = 'portfolio.db'  # Change to your actual DB filename if different

def print_table(conn, table):
    print(f'--- {table} ---')
    for row in conn.execute(f'SELECT * FROM {table} LIMIT 10'):
        print(row)
    print()

def insert_sample_data(conn):
    # Insert sample quantity and price history for all real assets in the DB
    today = datetime.today()
    # Get all assets (id, ticker, quantity)
    assets = list(conn.execute("SELECT id, ticker, quantity FROM assets"))
    for asset_id, ticker, quantity in assets:
        # Insert price history for last 30 days for this ticker
        for i in range(30):
            d = today - timedelta(days=29-i)
            date_str = d.strftime('%Y-%m-%d')
            price = 100 + i  # Sample price trend, can be customized
            conn.execute("""
                INSERT OR IGNORE INTO daily_prices (ticker, date, close_price)
                VALUES (?, ?, ?)
            """, (ticker, date_str, price))
        # Insert quantity history for last 5 years for this asset_id, only if not already present
        for i in range(365*5):
            d = today - timedelta(days=(365*5-1-i))
            date_str = d.strftime('%Y-%m-%d')
            # Only insert if not already present for this asset_id and date
            exists = conn.execute(
                "SELECT 1 FROM asset_quantity_history WHERE asset_id = ? AND date = ?",
                (asset_id, date_str)
            ).fetchone()
            if not exists:
                qty = quantity if quantity is not None else 10
                conn.execute("""
                    INSERT INTO asset_quantity_history (asset_id, date, quantity)
                    VALUES (?, ?, ?)
                """, (asset_id, date_str, qty))
    conn.commit()
    print('Sample data inserted for all assets (5 years quantity history, current quantity used for all past days).')

def main():
    conn = sqlite3.connect(DB_FILENAME)
    print_table(conn, 'portfolios')
    print_table(conn, 'assets')
    print_table(conn, 'daily_prices')
    print_table(conn, 'asset_quantity_history')
    insert_sample_data(conn)
    print_table(conn, 'portfolios')
    print_table(conn, 'assets')
    print_table(conn, 'daily_prices')
    print_table(conn, 'asset_quantity_history')
    conn.close()

if __name__ == '__main__':
    main()
