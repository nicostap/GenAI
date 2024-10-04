
import requests
import yfinance as yf
import pandas as pd
import config
from ta import (trend, momentum, volatility, volume)

# import matplotlib.plot as plt
### **New Methods for Technical and Fundamental Analysis**

class CustomTools: 
    def get_stock_data(self, ticker, period="1mo"):
        """Fetch historical stock data from yfinance."""
        stock = yf.Ticker(ticker)
        data = stock.history(period=period)
        return data
    
    def analyze_stock(self, ticker: str, period="6mo"):
        """Perform comprehensive technical analysis on the stock."""
        data = self.get_stock_data(ticker, period)
        
        # Ensure there are enough data points
        if data.empty or len(data) < 50:
            return "Not enough data to perform analysis."
        
        # Calculate Technical Indicators
        try:
            # Trend Indicators
            data['SMA_20'] = trend.SMAIndicator(data['Close'], window=20).sma_indicator()
            data['EMA_20'] = trend.EMAIndicator(data['Close'], window=20).ema_indicator()
            data['MACD'] = trend.MACD(data['Close']).macd()
            data['MACD_Signal'] = trend.MACD(data['Close']).macd_signal()
            data['MACD_Diff'] = trend.MACD(data['Close']).macd_diff()
            
            # Momentum Indicators
            data['RSI'] = momentum.RSIIndicator(data['Close'], window=14).rsi()
            data['Stochastic'] = momentum.StochasticOscillator(data['High'], data['Low'], data['Close'], window=14).stoch()
            
            # Volatility Indicators
            data['Bollinger_High'] = volatility.BollingerBands(data['Close'], window=20).bollinger_hband()
            data['Bollinger_Low'] = volatility.BollingerBands(data['Close'], window=20).bollinger_lband()
            data['ATR'] = volatility.AverageTrueRange(data['High'], data['Low'], data['Close'], window=14).average_true_range()
            
            # Volume Indicators
            data['On_Balance_Volume'] = volume.OnBalanceVolumeIndicator(data['Close'], data['Volume']).on_balance_volume()
            
        except Exception as e:
            return f"Error calculating technical indicators: {e}"
        
        # Drop rows with NaN values
        data.dropna(inplace=True)
        
        # Example Analysis Summary
        latest = data.iloc[-1]
        analysis = f"""
        **Technical Analysis for {ticker.upper()}**
        
        - **SMA 20:** {latest['SMA_20']:.2f}
        - **EMA 20:** {latest['EMA_20']:.2f}
        - **MACD:** {latest['MACD']:.2f}
        - **MACD Signal:** {latest['MACD_Signal']:.2f}
        - **MACD Difference:** {latest['MACD_Diff']:.2f}
        - **RSI:** {latest['RSI']:.2f} ({'Overbought' if latest['RSI'] > 70 else 'Oversold' if latest['RSI'] < 30 else 'Neutral'})
        - **Stochastic Oscillator:** {latest['Stochastic']:.2f}
        - **Bollinger Bands:** High = {latest['Bollinger_High']:.2f}, Low = {latest['Bollinger_Low']:.2f}
        - **ATR:** {latest['ATR']:.2f}
        - **On-Balance Volume:** {latest['On_Balance_Volume']:.2f}
        """
        
        # Generate Plot
        plot = self.plot_technical_indicators(data, ticker)
        
        # Save plot to a temporary buffer
        import io
        buf = io.BytesIO()
        plot.savefig(buf, format="png")
        buf.seek(0)
        plot_image = buf.read()
        
        # Encode plot image to base64
        plot_base64 = base64.b64encode(plot_image).decode()
        
        analysis += f"\n![Technical Indicators](data:image/png;base64,{plot_base64})"
        
        return analysis
    
    def plot_technical_indicators(self, data, ticker):
        """Plot technical indicators."""
        plt.figure(figsize=(14, 7))
        plt.plot(data['Close'], label='Close Price', color='blue')
        plt.plot(data['SMA_20'], label='SMA 20', color='red', linestyle='--')
        plt.plot(data['EMA_20'], label='EMA 20', color='green', linestyle='--')
        plt.fill_between(data.index, data['Bollinger_High'], data['Bollinger_Low'], color='grey', alpha=0.1, label='Bollinger Bands')
        plt.title(f"Technical Indicators for {ticker.upper()}")
        plt.legend()
        plt.tight_layout()
        return plt
    
    def analyze_fundamental(self, ticker):
        """Perform fundamental analysis on the stock."""
        stock = yf.Ticker(ticker)
        
        try:
            info = stock.info
        except Exception as e:
            return f"Error fetching fundamental data: {e}"
        
        # Extract Key Fundamental Metrics
        metrics = {
            "Market Cap": f"${info.get('marketCap', 'N/A'):,}" if info.get('marketCap') else "N/A",
            "PE Ratio": info.get("trailingPE", "N/A"),
            "EPS": info.get("trailingEps", "N/A"),
            "Revenue Growth": f"{info.get('revenueGrowth', 'N/A')*100:.2f}%" if info.get('revenueGrowth') else "N/A",
            "Profit Margins": f"{info.get('profitMargins', 'N/A')*100:.2f}%" if info.get('profitMargins') else "N/A",
            "Return on Equity": f"{info.get('returnOnEquity', 'N/A')*100:.2f}%" if info.get('returnOnEquity') else "N/A",
            "Debt to Equity": info.get("debtToEquity", "N/A"),
            "Beta": info.get("beta", "N/A"),
            "52-Week Change": f"{info.get('52WeekChange', 'N/A')*100:.2f}%" if info.get('52WeekChange') else "N/A",
            "Average Volume": f"{info.get('averageVolume', 'N/A'):,}" if info.get('averageVolume') else "N/A",
        }
        
        # Format Metrics
        metrics_formatted = "\n".join([f"- **{key}:** {value}" for key, value in metrics.items()])
        
        analysis = f"""
        **Fundamental Analysis for {ticker.upper()}**
        
        {metrics_formatted}
        """
        
        return analysis
    
    def extract_ticker(self, prompt):
        """Extract the ticker symbol from the user's prompt."""
        # This is a simplistic approach; consider using regex or NLP for better extraction
        words = prompt.split()
        tickers = [word.upper() for word in words if word.isupper() and len(word) <= 5]
        return tickers[0] if tickers else None
    
    def validate_ticker(self, ticker):
        """Validate if the ticker exists."""
        try:
            stock = yf.Ticker(ticker)
            # Check if the info dictionary is populated
            return stock.info and 'regularMarketPrice' in stock.info
        except:
            return False
