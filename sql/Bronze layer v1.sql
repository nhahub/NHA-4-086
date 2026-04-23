
CREATE TABLE bronze.stocks_info (
    symbol TEXT,
    name TEXT,
    price_intraday TEXT,
    change TEXT,
    change_percent TEXT,
    volume TEXT,
    avg_vol_3m TEXT,
    market_cap TEXT,
    pe_ratio_ttm TEXT,
    week_52_range TEXT,
    region TEXT,
    sector TEXT,
    industry TEXT,
    index_info TEXT,
    category TEXT,
    extraction_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);


CREATE TABLE bronze.stock_prices_raw (
    date TEXT,
    close TEXT,
    high TEXT,
    low TEXT,
    open TEXT,
    volume TEXT,
    ticker TEXT,
    category TEXT,
    extraction_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
