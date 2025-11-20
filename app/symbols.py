"""Symbol definitions for trading data pipeline."""
# For testing purpose
# DAILY_BAR_SYMBOLS = ["AAPL"]   # Apple
# INTRADAY_BAR_SYMBOLS = ["AAPL"]   # Apple
# NEWS_SYMBOLS = ["AAPL"]   # Apple

# Market index definitions
# Polygon.io uses "I:" prefix for indices
MARKET_INDICES = {
    # "US": "I:SPX",  # S&P 500 Index (primary US market index)
    "US": "SPY",  # S&P 500 Index (primary US market index)
    "SPY": "SPY",   # SPDR S&P 500 ETF (alternative proxy)
}

# Market index mapping for each symbol
# Most symbols are US stocks, so they use S&P 500
# International stocks (BABA, TM, TSM, HSBC) also default to S&P 500 for now
# Note: For more accurate beta, international stocks should use their respective regional indices
SYMBOL_TO_MARKET_INDEX = {
    # US stocks - all use S&P 500
    "AAPL": MARKET_INDICES["US"],
    "MSFT": MARKET_INDICES["US"],
    "GOOGL": MARKET_INDICES["US"],
    "AMZN": MARKET_INDICES["US"],
    "META": MARKET_INDICES["US"],
    "NVDA": MARKET_INDICES["US"],
    "INTC": MARKET_INDICES["US"],
    "CSCO": MARKET_INDICES["US"],
    "ORCL": MARKET_INDICES["US"],
    "IBM": MARKET_INDICES["US"],
    "JPM": MARKET_INDICES["US"],
    "BAC": MARKET_INDICES["US"],
    "WFC": MARKET_INDICES["US"],
    "GS": MARKET_INDICES["US"],
    "MS": MARKET_INDICES["US"],
    "V": MARKET_INDICES["US"],
    "MA": MARKET_INDICES["US"],
    "AXP": MARKET_INDICES["US"],
    "JNJ": MARKET_INDICES["US"],
    "PFE": MARKET_INDICES["US"],
    "MRK": MARKET_INDICES["US"],
    "ABBV": MARKET_INDICES["US"],
    "LLY": MARKET_INDICES["US"],
    "XOM": MARKET_INDICES["US"],
    "CVX": MARKET_INDICES["US"],
    "COP": MARKET_INDICES["US"],
    "NEE": MARKET_INDICES["US"],
    "WMT": MARKET_INDICES["US"],
    "COST": MARKET_INDICES["US"],
    "TGT": MARKET_INDICES["US"],
    "HD": MARKET_INDICES["US"],
    "PG": MARKET_INDICES["US"],
    "KO": MARKET_INDICES["US"],
    "PEP": MARKET_INDICES["US"],
    "TSLA": MARKET_INDICES["US"],
    "F": MARKET_INDICES["US"],
    "GM": MARKET_INDICES["US"],
    "CAT": MARKET_INDICES["US"],
    "BA": MARKET_INDICES["US"],
    "CMCSA": MARKET_INDICES["US"],
    "DIS": MARKET_INDICES["US"],
    "NFLX": MARKET_INDICES["US"],
    "T": MARKET_INDICES["US"],
    # International stocks - defaulting to S&P 500 for now
    # TODO: Map to appropriate regional indices (e.g., BABA -> I:SHCOMP, TM -> I:N225, etc.)
    "BABA": MARKET_INDICES["US"],  # Alibaba (China ADR)
    "TM": MARKET_INDICES["US"],    # Toyota (Japan ADR)
    "TSM": MARKET_INDICES["US"],   # Taiwan Semiconductor (Taiwan ADR)
    "HSBC": MARKET_INDICES["US"],  # HSBC Holdings (UK ADR)
}

def get_market_index(ticker: str) -> str:
    """
    Get the market index symbol for a given ticker.
    
    Args:
        ticker: Stock ticker symbol
        
    Returns:
        Market index symbol (e.g., "I:SPX" for S&P 500)
    """
    return SYMBOL_TO_MARKET_INDEX.get(ticker, MARKET_INDICES["US"])

# Symbols for daily bars backfill (all major symbols)
DAILY_BAR_SYMBOLS = [
    # Technology
    "AAPL",   # Apple
    "MSFT",   # Microsoft
    "GOOGL",  # Alphabet (Google)
    "AMZN",   # Amazon
    "META",   # Meta (Facebook)
    "NVDA",   # Nvidia
    "INTC",   # Intel
    "CSCO",   # Cisco
    "ORCL",   # Oracle
    "IBM",    # IBM
    
    # Financial
    "JPM",    # JPMorgan Chase
    "BAC",    # Bank of America
    "WFC",    # Wells Fargo
    "GS",     # Goldman Sachs
    "MS",     # Morgan Stanley
    "V",      # Visa
    "MA",     # Mastercard
    "AXP",    # American Express
    
    # Healthcare
    "JNJ",    # Johnson & Johnson
    "PFE",    # Pfizer
    "MRK",    # Merck & Co.
    "ABBV",   # AbbVie
    "LLY",    # Eli Lilly
    
    # Energy
    "XOM",    # ExxonMobil
    "CVX",    # Chevron
    "COP",    # ConocoPhillips
    "NEE",    # NextEra Energy
    
    # Consumer
    "WMT",    # Walmart
    "COST",   # Costco
    "TGT",    # Target
    "HD",     # Home Depot
    "PG",     # Procter & Gamble
    "KO",     # Coca-Cola
    "PEP",    # PepsiCo
    
    # Industrial/Auto
    "TSLA",   # Tesla
    "F",      # Ford
    "GM",     # General Motors
    "CAT",    # Caterpillar
    "BA",     # Boeing
    
    # Media/Communication
    "CMCSA",  # Comcast
    "DIS",    # Disney
    "NFLX",   # Netflix
    "T",      # AT&T
    
    # International
    "BABA",   # Alibaba
    "TM",     # Toyota
    "TSM",    # Taiwan Semiconductor
    "HSBC"    # HSBC Holdings
]
    
INTRADAY_BAR_SYMBOLS = [
# Technology
    "AAPL",   # Apple
    "MSFT",   # Microsoft
    "GOOGL",  # Alphabet (Google)
    "AMZN",   # Amazon
    "META",   # Meta (Facebook)
    "NVDA",   # Nvidia
    "INTC",   # Intel
    "CSCO",   # Cisco
    "ORCL",   # Oracle
    "IBM",    # IBM
    
    # Financial
    "JPM",    # JPMorgan Chase
    "BAC",    # Bank of America
    "WFC",    # Wells Fargo
    "GS",     # Goldman Sachs
    "MS",     # Morgan Stanley
    "V",      # Visa
    "MA",     # Mastercard
    "AXP",    # American Express
    
    # Healthcare
    "JNJ",    # Johnson & Johnson
    "PFE",    # Pfizer
    "MRK",    # Merck & Co.
    "ABBV",   # AbbVie
    "LLY",    # Eli Lilly
    
    # Energy
    "XOM",    # ExxonMobil
    "CVX",    # Chevron
    "COP",    # ConocoPhillips
    "NEE",    # NextEra Energy
    
    # Consumer
    "WMT",    # Walmart
    "COST",   # Costco
    "TGT",    # Target
    "HD",     # Home Depot
    "PG",     # Procter & Gamble
    "KO",     # Coca-Cola
    "PEP",    # PepsiCo
    
    # Industrial/Auto
    "TSLA",   # Tesla
    "F",      # Ford
    "GM",     # General Motors
    "CAT",    # Caterpillar
    "BA",     # Boeing
    
    # Media/Communication
    "CMCSA",  # Comcast
    "DIS",    # Disney
    "NFLX",   # Netflix
    "T",      # AT&T
    
    # International
    "BABA",   # Alibaba
    "TM",     # Toyota
    "TSM",    # Taiwan Semiconductor
    "HSBC"    # HSBC Holdings
]

NEWS_SYMBOLS = [
# Technology
    "AAPL",   # Apple
    "MSFT",   # Microsoft
    "GOOGL",  # Alphabet (Google)
    "AMZN",   # Amazon
    "META",   # Meta (Facebook)
    "NVDA",   # Nvidia
    "INTC",   # Intel
    "CSCO",   # Cisco
    "ORCL",   # Oracle
    "IBM",    # IBM
    
    # Financial
    "JPM",    # JPMorgan Chase
    "BAC",    # Bank of America
    "WFC",    # Wells Fargo
    "GS",     # Goldman Sachs
    "MS",     # Morgan Stanley
    "V",      # Visa
    "MA",     # Mastercard
    "AXP",    # American Express
    
    # Healthcare
    "JNJ",    # Johnson & Johnson
    "PFE",    # Pfizer
    "MRK",    # Merck & Co.
    "ABBV",   # AbbVie
    "LLY",    # Eli Lilly
    
    # Energy
    "XOM",    # ExxonMobil
    "CVX",    # Chevron
    "COP",    # ConocoPhillips
    "NEE",    # NextEra Energy
    
    # Consumer
    "WMT",    # Walmart
    "COST",   # Costco
    "TGT",    # Target
    "HD",     # Home Depot
    "PG",     # Procter & Gamble
    "KO",     # Coca-Cola
    "PEP",    # PepsiCo
    
    # Industrial/Auto
    "TSLA",   # Tesla
    "F",      # Ford
    "GM",     # General Motors
    "CAT",    # Caterpillar
    "BA",     # Boeing
    
    # Media/Communication
    "CMCSA",  # Comcast
    "DIS",    # Disney
    "NFLX",   # Netflix
    "T",      # AT&T
    
    # International
    "BABA",   # Alibaba
    "TM",     # Toyota
    "TSM",    # Taiwan Semiconductor
    "HSBC"    # HSBC Holdings
]