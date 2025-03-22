import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import datetime
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.dates as mdates
from datetime import timedelta

# Set random seeds for reproducibility

class StockLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, dropout=0.3):
        super(StockLSTM, self).__init__()
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True  # Use bidirectional LSTM for better context
        )
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # *2 because of bidirectional
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            nn.Softmax(dim=1)
        )
        
        # Fully connected layers for regression
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)  # *2 because of bidirectional
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.bn2 = nn.BatchNorm1d(hidden_dim // 2)
        self.dropout2 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(hidden_dim // 2, 1)  # Output is a single value (percentage change)
        
        self.leaky_relu = nn.LeakyReLU(0.2)
    
    def forward(self, x):
        # Pass through LSTM
        lstm_out, _ = self.lstm(x)  # Shape: (batch_size, seq_len, hidden_dim*2)
        
        # Apply attention to the LSTM outputs
        attention_weights = self.attention(lstm_out)  # Shape: (batch_size, seq_len, 1)
        context = torch.sum(attention_weights * lstm_out, dim=1)  # Shape: (batch_size, hidden_dim*2)
        
        # Fully connected layers
        x = self.fc1(context)
        x = self.bn1(x)
        x = self.leaky_relu(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.leaky_relu(x)
        x = self.dropout2(x)
        
        x = self.fc3(x)  # Final output is percentage change prediction
        
        return x

def calculate_technical_indicators(df):
    """Calculate various technical indicators."""
    # Make a copy to avoid warnings
    data = df.copy()
    
    # Simple Moving Averages
    for window in [5, 10, 20, 50]:
        data[f'SMA_{window}'] = data['Close'].rolling(window=window).mean()
    
    # Exponential Moving Averages
    for window in [5, 10, 20, 50]:
        data[f'EMA_{window}'] = data['Close'].ewm(span=window, adjust=False).mean()
    
    # Moving Average Convergence Divergence (MACD)
    data['MACD'] = data['Close'].ewm(span=12, adjust=False).mean() - \
                  data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD_Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
    data['MACD_Hist'] = data['MACD'] - data['MACD_Signal']
    
    # Relative Strength Index (RSI)
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    
    rs = avg_gain / avg_loss
    data['RSI'] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands
    data['BB_Middle'] = data['Close'].rolling(window=20).mean()
    data['BB_Std'] = data['Close'].rolling(window=20).std()
    data['BB_Upper'] = data['BB_Middle'] + 2 * data['BB_Std']
    data['BB_Lower'] = data['BB_Middle'] - 2 * data['BB_Std']
    data['BB_Width'] = (data['BB_Upper'] - data['BB_Lower']) / data['BB_Middle']
    data['BB_Position'] = (data['Close'] - data['BB_Lower']) / (data['BB_Upper'] - data['BB_Lower'])
    
    # Price-to-SMA ratios (momentum indicators)
    for window in [10, 20, 50]:
        data[f'Price_to_SMA_{window}'] = data['Close'] / data[f'SMA_{window}']
    
    # Volume Indicators
    data['Volume_SMA_5'] = data['Volume'].rolling(window=5).mean()
    data['Volume_SMA_20'] = data['Volume'].rolling(window=20).mean()
    data['Volume_Ratio'] = data['Volume'] / data['Volume_SMA_5']
    data['OBV'] = (np.sign(data['Close'].diff()) * data['Volume']).fillna(0).cumsum()
    
    # Rate of Change
    for window in [1, 5, 10, 20]:
        data[f'ROC_{window}'] = data['Close'].pct_change(periods=window) * 100
    
    # Average True Range (ATR)
    high_low = data['High'] - data['Low']
    high_close = (data['High'] - data['Close'].shift(1)).abs()
    low_close = (data['Low'] - data['Close'].shift(1)).abs()
    
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    data['ATR_14'] = true_range.rolling(window=14).mean()
    
    # Stochastic Oscillator
    low_14 = data['Low'].rolling(window=14).min()
    high_14 = data['High'].rolling(window=14).max()
    data['%K'] = 100 * ((data['Close'] - low_14) / (high_14 - low_14))
    data['%D'] = data['%K'].rolling(window=3).mean()
    
    # Daily Returns and Volatility
    data['Daily_Return'] = data['Close'].pct_change() * 100
    data['Return_Volatility_14'] = data['Daily_Return'].rolling(window=14).std()
    data['Return_Volatility_30'] = data['Daily_Return'].rolling(window=30).std()
    
    # High/Low indicators
    data['HL_Ratio'] = data['High'] / data['Low']
    data['HC_Ratio'] = data['High'] / data['Close'] 
    data['LC_Ratio'] = data['Low'] / data['Close']
    
    # Momentum Indicators
    data['Momentum_5'] = data['Close'] / data['Close'].shift(5) - 1
    data['Momentum_10'] = data['Close'] / data['Close'].shift(10) - 1
    data['Momentum_20'] = data['Close'] / data['Close'].shift(20) - 1
    
    # Gap indicators
    data['Gap'] = data['Open'] / data['Close'].shift(1) - 1
    data['Gap_MA_5'] = data['Gap'].rolling(window=5).mean()
    
    return data

def prepare_data(ticker='TSLA', start_date='2015-01-01', end_date=None, seq_length=20, train_ratio=0.8, target_type='pct_change'):
    """
    Prepare data for LSTM model with technical indicators.
    
    Args:
        ticker: Stock ticker symbol
        start_date: Start date for data retrieval
        end_date: End date for data retrieval (default: today)
        seq_length: Number of time steps for LSTM input
        train_ratio: Ratio of data to use for training
        target_type: 'pct_change' (percentage change) or 'price' (actual price)
        
    Returns:
        Tuple of (X_train, y_train, X_test, y_test, scalers, test_dates)
    """
    # Set end date if not provided
    if end_date is None:
        end_date = datetime.date.today().strftime('%Y-%m-%d')
    
    # Fetch the data
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    
    # Fix multi-index columns from yfinance
    stock_data.columns = stock_data.columns.get_level_values(0)
    
    # Keep track of original data for visualization
    original_data = stock_data.copy()
    
    # Get SP500 data for comparison
    sp500_data = yf.download('^GSPC', start=start_date, end=end_date)
    
    # Fix multi-index columns for SP500 data
    sp500_data.columns = sp500_data.columns.get_level_values(0)
    
    # Add SP500 to stock data
    stock_data['SP500_Close'] = sp500_data['Close']
    stock_data['SP500_Return'] = sp500_data['Close'].pct_change() * 100
    
    # Calculate relative performance to market
    stock_data['Relative_Return'] = stock_data['Close'].pct_change() * 100 - stock_data['SP500_Return']
    stock_data['Market_Beta'] = stock_data['Close'].pct_change().rolling(60).cov(
        sp500_data['Close'].pct_change()) / sp500_data['Close'].pct_change().rolling(60).var()
    
    # Calculate technical indicators
    data = calculate_technical_indicators(stock_data)
    
    # Set target variable with larger window options
    if target_type == 'pct_change':
        # Percentage change for next day with more options
        data['Target'] = data['Close'].pct_change(1).shift(-1) * 100  # Next day's percentage change
        
        # Add different target windows for potentially better signal
        data['Target_2day'] = data['Close'].pct_change(2).shift(-2) * 100  # 2-day forward change
        data['Target_5day'] = data['Close'].pct_change(5).shift(-5) * 100  # 5-day forward change
        
        # We'll stick with 'Target' as our main target
    else:
        # Actual price for next day
        data['Target'] = data['Close'].shift(-1)  # Next day's closing price
    
    # Drop NaN values
    data.dropna(inplace=True)
    
    # Define features (exclude target and close price to avoid leakage)
    price_columns = ['Open', 'High', 'Low', 'Close', 'Adj Close']
    target_columns = ['Target', 'Target_2day', 'Target_5day'] if 'Target_2day' in data.columns else ['Target']
    feature_cols = [col for col in data.columns if col not in price_columns + target_columns]
    
    # Split data into features and target
    X = data[feature_cols].values
    y = data['Target'].values.reshape(-1, 1)
    
    # Save feature column names for later use
    feature_names = data[feature_cols].columns.tolist()
    
    # Split into train and test sets (80% train, 20% test or as specified)
    train_size = int(len(X) * train_ratio)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    test_dates = data.index[train_size:]
    
    # Store original close prices for later visualization
    close_train = data['Close'].values[:train_size]
    close_test = data['Close'].values[train_size:]
    
    # Scale features
    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)
    
    # For percentage targets, we might want to scale them too if they have large variance
    if target_type == 'pct_change':
        # Preserve the sign but scale the magnitude
        y_train_sign = np.sign(y_train)
        y_test_sign = np.sign(y_test)
        
        y_train_abs = np.abs(y_train)
        y_test_abs = np.abs(y_test)
        
        # Scale the absolute values
        scaler_y = MinMaxScaler(feature_range=(0, 1))
        y_train_abs_scaled = scaler_y.fit_transform(y_train_abs)
        y_test_abs_scaled = scaler_y.transform(y_test_abs)
        
        # Restore the signs
        y_train_scaled = y_train_sign * y_train_abs_scaled
        y_test_scaled = y_test_sign * y_test_abs_scaled
    else:
        # Scale price targets
        scaler_y = StandardScaler()
        y_train_scaled = scaler_y.fit_transform(y_train)
        y_test_scaled = scaler_y.transform(y_test)
    
    # Create sequences for LSTM
    def create_sequences(X, y, seq_length):
        X_seq, y_seq = [], []
        for i in range(len(X) - seq_length):
            X_seq.append(X[i:i+seq_length])
            y_seq.append(y[i+seq_length])
        return np.array(X_seq), np.array(y_seq)
    
    X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train_scaled, seq_length)
    X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test_scaled, seq_length)
    
    # Adjust test_dates to match sequence length
    test_dates = test_dates[seq_length:]
    close_test = close_test[seq_length:]
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train_seq, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_seq, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test_seq, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test_seq, dtype=torch.float32)
    
    # Create data loaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False)
    
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    return (X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, 
            train_loader, test_loader, scaler_X, scaler_y, test_dates, close_test,
            original_data, feature_names)

def create_performance_summary_chart(investment_metrics, ticker):
    """
    Create a comprehensive performance summary chart with metrics.
    
    Args:
        investment_metrics: Dictionary with investment performance metrics from simulation
        ticker: Stock ticker symbol
    """
    df = investment_metrics['Data']
    investment_date = df['Date'].iloc[0].strftime('%Y-%m-%d')
    end_date = df['Date'].iloc[-1].strftime('%Y-%m-%d')
    
    # Create figure and primary axis
    fig, ax1 = plt.subplots(figsize=(16, 10))
    
    # Plot price chart
    ax1.plot(df['Date'], df['Close'], color='gray', alpha=0.5, linewidth=1)
    
    # Plot buy/sell signals
    buy_signals = df[df['Signal'] == 1]
    ax1.scatter(buy_signals['Date'], buy_signals['Close'], marker='^', color='green', s=100, label='Buy Signal')
    
    sell_signals = df[df['Signal'] == -1]
    ax1.scatter(sell_signals['Date'], sell_signals['Close'], marker='v', color='red', s=100, label='Sell Signal')
    
    # Plot investment values
    ax2 = ax1.twinx()
    ax2.plot(df['Date'], df['Buy_Hold_Equity'], label='Buy & Hold Strategy', color='blue', linewidth=2)
    ax2.plot(df['Date'], df['Strategy_Equity'], label='Model Strategy', color='green', linewidth=2)
    
    # Add key metrics as text box
    metrics_text = (
        f"Initial Investment: ${investment_metrics['Initial_Investment']:.2f}\n"
        f"Initial Price: ${investment_metrics['Initial_Price']:.2f}\n"
        f"Final Price: ${investment_metrics['Final_Price']:.2f}\n"
        f"Price Change: {((investment_metrics['Final_Price']/investment_metrics['Initial_Price'])-1)*100:.2f}%\n\n"
        f"Buy & Hold Return: {investment_metrics['Buy_Hold_Return']:.2f}%\n"
        f"Model Strategy Return: {investment_metrics['Strategy_Return']:.2f}%\n"
        f"Outperformance: {investment_metrics['Outperformance']:.2f}%\n\n"
        f"Total Trades: {investment_metrics['Total_Trades']}\n"
        f"Direction Accuracy: {investment_metrics['Direction_Accuracy']:.2f}%"
    )
    
    # Add text box for metrics
    props = dict(boxstyle='round', facecolor='white', alpha=0.7)
    ax1.text(1.02, 0.5, metrics_text, transform=ax1.transAxes, fontsize=12,
             verticalalignment='center', bbox=props)
    
    # Set labels and title
    ax1.set_ylabel('Price ($)', fontsize=12)
    ax2.set_ylabel('Portfolio Value ($)', fontsize=12)
    ax1.set_title(f'{ticker} - Investment Performance ({investment_date} to {end_date})', fontsize=16)
    
    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=12)
    
    # Format x-axis
    plt.xticks(rotation=45)
    fig.tight_layout()
    plt.savefig(f'{ticker}_consolidated_performance_{investment_date.replace("-", "")}.png')
    plt.close()

def train_model(model, train_loader, X_val, y_val, num_epochs=300, patience=30):
    """
    Train the LSTM model with improved training strategy.
    
    Args:
        model: PyTorch model
        train_loader: DataLoader for training data
        X_val: Validation features tensor
        y_val: Validation target tensor
        num_epochs: Maximum number of epochs
        patience: Early stopping patience
        
    Returns:
        Trained model and training history
    """
    # Loss function: use MSE for regression
    criterion = nn.HuberLoss()
    
    # Optimizer: Adam with weight decay for regularization
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, verbose=True
    )
    
    # Early stopping parameters
    best_val_loss = float('inf')
    early_stop_counter = 0
    best_model_state = None
    
    # Training history
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            val_loss = criterion(val_outputs, y_val).item()
        
        val_losses.append(val_loss)
        scheduler.step(val_loss)
        
        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}/{num_epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f}')
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            
        if early_stop_counter >= patience:
            print(f'Early stopping at epoch {epoch+1}')
            break
    
    # Load best model
    model.load_state_dict(best_model_state)
    
    # Plot training history
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig('training_loss.png')
    
    return model, (train_losses, val_losses)

def evaluate_model(model, X_test, y_test, test_dates, close_prices, original_data, ticker, scaler_y=None, target_type='pct_change'):
    """
    Evaluate the model performance and create detailed visualizations.
    
    Args:
        model: Trained PyTorch model
        X_test: Test features tensor
        y_test: Test target tensor
        test_dates: Dates for test data
        close_prices: Close prices for test data
        original_data: Original stock data
        scaler_y: Scaler for target variable
        target_type: 'pct_change' or 'price'
    
    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()
    
    # Make predictions
    with torch.no_grad():
        predictions = model(X_test).numpy()
    
    # Get actual values
    actuals = y_test.numpy()
    
    # If we scaled the target values, inverse transform them
    if scaler_y is not None:
        if target_type == 'pct_change':
            # For percentage change, we scaled magnitude but preserved sign
            predictions_sign = np.sign(predictions)
            actuals_sign = np.sign(actuals)
            
            predictions_abs = np.abs(predictions)
            actuals_abs = np.abs(actuals)
            
            predictions_abs = scaler_y.inverse_transform(predictions_abs)
            actuals_abs = scaler_y.inverse_transform(actuals_abs)
            
            predictions = predictions_sign * predictions_abs
            actuals = actuals_sign * actuals_abs
        else:
            # For price, we just scaled directly
            predictions = scaler_y.inverse_transform(predictions)
            actuals = scaler_y.inverse_transform(actuals)
    
    # Calculate metrics
    mse = mean_squared_error(actuals, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actuals, predictions)
    r2 = r2_score(actuals, predictions)
    
    # Calculate directional accuracy
    direction_correct = np.sum((predictions > 0) == (actuals > 0))
    direction_accuracy = direction_correct / len(predictions) * 100
    
    # Print metrics
    print(f"Evaluation Metrics for {ticker}:")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R² Score: {r2:.4f}")
    print(f"Directional Accuracy: {direction_accuracy:.2f}%")
    
    # Prepare data for visualizations
    dates = test_dates
    
    # Convert percentage predictions to prices for visualization
    if target_type == 'pct_change':
        # Calculate predicted and actual prices
        predicted_prices = []
        actual_prices = []
        
        for i in range(len(predictions)):
            if i == 0:
                # First prediction requires the last close price from training
                last_price = close_prices[0]
            else:
                last_price = close_prices[i]
            
            actual_return = actuals[i][0] / 100  # Convert from percentage to decimal
            predicted_return = predictions[i][0] / 100
            
            actual_price = last_price * (1 + actual_return)
            predicted_price = last_price * (1 + predicted_return)
            
            actual_prices.append(actual_price)
            predicted_prices.append(predicted_price)
        
        predicted_prices = np.array(predicted_prices)
        actual_prices = np.array(actual_prices)
    else:
        # For direct price prediction
        predicted_prices = predictions
        actual_prices = actuals
    
    # Create visualization for percentage change
    plt.figure(figsize=(16, 8))
    plt.subplot(2, 1, 1)
    plt.plot(dates, actuals, label=f'{ticker} Actual % Change', color='blue', alpha=0.7)
    plt.plot(dates, predictions, label='Predicted % Change', color='red', linestyle='-')
    plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    plt.title(f'{ticker} Daily Percentage Change: Actual vs Predicted', fontsize=14)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Percentage Change', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(alpha=0.3)
    
    # Format x-axis dates
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45)
    
    # Create visualization for price comparison
    plt.subplot(2, 1, 2)
    plt.plot(dates, actual_prices, label='Actual Close Price', color='green', alpha=0.7)
    plt.plot(dates, predicted_prices, label='Predicted Close Price', color='orange', linestyle='-')
    plt.title(f'{ticker} Stock Price: Actual vs Predicted', fontsize=14)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Price ($)', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(alpha=0.3)
    
    # Format x-axis dates
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(f'{ticker}_prediction_comparison.png')
    
    # Create a zoomed view of the last 60 days for better visibility
    last_n_days = min(60, len(dates))
    
    plt.figure(figsize=(16, 8))
    plt.subplot(2, 1, 1)
    plt.plot(dates[-last_n_days:], actuals[-last_n_days:], label='Actual % Change', color='blue', alpha=0.7)
    plt.plot(dates[-last_n_days:], predictions[-last_n_days:], label='Predicted % Change', color='red', linestyle='-')
    plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    plt.title(f'{ticker} Last {last_n_days} Days - Daily Percentage Change: Actual vs Predicted', fontsize=14)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Percentage Change', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(alpha=0.3)
    
    # Format x-axis dates
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    plt.xticks(rotation=45)
    
    plt.subplot(2, 1, 2)
    plt.plot(dates[-last_n_days:], actual_prices[-last_n_days:], label='Actual Close Price', color='green', alpha=0.7)
    plt.plot(dates[-last_n_days:], predicted_prices[-last_n_days:], label='Predicted Close Price', color='orange', linestyle='-')
    plt.title(f'{ticker} Last {last_n_days} Days - Stock Price: Actual vs Predicted', fontsize=14)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Price ($)', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(alpha=0.3)
    
    # Format x-axis dates
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(f'{ticker}_prediction_comparison_recent.png')
    
    # Create scatter plot of predicted vs actual
    plt.figure(figsize=(10, 8))
    
    plt.scatter(actuals, predictions, alpha=0.5)
    min_val = min(np.min(actuals), np.min(predictions))
    max_val = max(np.max(actuals), np.max(predictions))
    plt.plot([min_val, max_val], [min_val, max_val], 'k--')
    plt.xlabel('Actual % Change', fontsize=12)
    plt.ylabel('Predicted % Change', fontsize=12)
    plt.title(f'{ticker} Predicted vs Actual Percentage Change', fontsize=14)
    plt.grid(alpha=0.3)
    plt.savefig(f'{ticker}_predicted_vs_actual_scatter.png')
    
    # Return evaluation metrics
    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2,
        'Direction_Accuracy': direction_accuracy,
        'Predictions': predictions,
        'Actuals': actuals,
        'Predicted_Prices': predicted_prices,
        'Actual_Prices': actual_prices
    }

def predict_next_day(model, scaler, data, features, sequence_length, target_scaler=None, target_type='price'):
    # Debug information
    print(f"Data in predict_next_day - shape: {data.shape if not isinstance(data, type(None)) else 'None'}")
    print(f"Data in predict_next_day - columns: {data.columns.tolist() if not isinstance(data, type(None)) and not data.empty else 'Empty'}")
    
    # Check if data is empty or None
    if isinstance(data, type(None)) or data.empty:
        print("ERROR: Data is empty or None in predict_next_day function")
        return None, None, None, None
    
    try:
        # Try to access the last element and print information
        print(f"Attempting to access last row of data with index {len(data)-1}")
        latest_close = data['Close'].iloc[-1]
        latest_date = data.index[-1]
        
        # Get next business day
        next_date = get_next_business_day(latest_date)
        
        # Prepare the sequence for prediction
        sequence = []
        for i in range(sequence_length):
            if i < len(data):
                row = data.iloc[len(data)-sequence_length+i][features].values
            else:
                # If we don't have enough data, use the last available row
                row = data.iloc[-1][features].values
            sequence.append(row)
        
        sequence = np.array(sequence)
        
        # Scale the sequence
        scaled_sequence = np.zeros_like(sequence)
        for i in range(len(features)):
            scaled_sequence[:, i] = scaler[features[i]].transform(sequence[:, i].reshape(-1, 1)).flatten()
        
        # Convert to tensor and predict
        X = torch.tensor(scaled_sequence).float().unsqueeze(0)
        model.eval()
        with torch.no_grad():
            prediction = model(X).item()
        
        # If we're predicting price directly
        if target_type == 'price':
            if target_scaler:
                prediction = target_scaler.inverse_transform([[prediction]])[0][0]
            next_day_price = prediction
            next_day_prediction = prediction
        
        # If we're predicting price change
        elif target_type == 'change':
            if target_scaler:
                prediction = target_scaler.inverse_transform([[prediction]])[0][0]
            next_day_price = latest_close + prediction
            next_day_prediction = prediction
        
        # If we're predicting percentage change
        elif target_type == 'percent_change':
            if target_scaler:
                prediction = target_scaler.inverse_transform([[prediction]])[0][0]
            next_day_price = latest_close * (1 + prediction / 100)
            next_day_prediction = prediction
        
        # If we're predicting direction
        elif target_type == 'direction':
            next_day_prediction = prediction
            if prediction > 0.5:
                # Positive direction: assume 1% increase for simplicity
                next_day_price = latest_close * 1.01
            else:
                # Negative direction: assume 1% decrease for simplicity
                next_day_price = latest_close * 0.99
        
        return next_day_prediction, next_day_price, latest_date, next_date
    
    except IndexError as e:
        print(f"IndexError: {e}")
        print(f"Data index length: {len(data.index) if not data.empty else 0}")
        print(f"Data columns: {data.columns.tolist() if not data.empty else []}")
        if not data.empty and 'Close' in data.columns:
            print(f"Close column length: {len(data['Close'])}")
        return None, None, None, None
    except Exception as e:
        print(f"Unexpected error in predict_next_day: {type(e).__name__}: {e}")
        return None, None, None, None
def predict_for_date(model, scaler_X, scaler_y, feature_cols, ticker='TSLA', 
                    date_str=None, seq_length=20, target_type='pct_change'):
    """
    Predict the percentage change for a specific historical date.
    
    Args:
        model: Trained PyTorch model
        scaler_X: Scaler for features
        scaler_y: Scaler for target (or None if no scaling needed)
        feature_cols: Feature column names
        ticker: Stock ticker symbol
        date_str: The date to predict for (format: 'YYYY-MM-DD')
        seq_length: Sequence length for the model
        target_type: 'pct_change' or 'price'
        
    Returns:
        Dictionary with prediction details and actual results
    """
    if date_str is None:
        raise ValueError("A date must be provided in 'YYYY-MM-DD' format")
    
    # Parse the target date
    target_date = datetime.datetime.strptime(date_str, '%Y-%m-%d').date()
    
    # Calculate the next trading day
    next_date = target_date + timedelta(days=1)
    while next_date.weekday() > 4:  # Skip weekends
        next_date = next_date + timedelta(days=1)
    
    # Calculate start date to get enough data
    start_date_obj = target_date - timedelta(days=seq_length*3 + 60)  # Extra days for indicators and to handle weekends
    start_date = start_date_obj.strftime('%Y-%m-%d')
    end_date = next_date + timedelta(days=5)  # Get a few days after to have actual results
    end_date = end_date.strftime('%Y-%m-%d')
    
    # Fetch the data
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    sp500_data = yf.download('^GSPC', start=start_date, end=end_date)
    
    # Fix multi-index columns from yfinance
    stock_data.columns = stock_data.columns.get_level_values(0)
    sp500_data.columns = sp500_data.columns.get_level_values(0)
    
    # Add SP500 to stock data
    stock_data['SP500_Close'] = sp500_data['Close']
    stock_data['SP500_Return'] = sp500_data['Close'].pct_change() * 100
    
    # Calculate relative performance to market
    stock_data['Relative_Return'] = stock_data['Close'].pct_change() * 100 - stock_data['SP500_Return']
    stock_data['Market_Beta'] = stock_data['Close'].pct_change().rolling(60).cov(
        sp500_data['Close'].pct_change()) / sp500_data['Close'].pct_change().rolling(60).var()
    
    # Calculate technical indicators
    data = calculate_technical_indicators(stock_data)
    
    # Drop NaN values
    data.dropna(inplace=True)
    
    # Find the exact date in data
    # Since market might be closed on target_date, get the closest date that's <= target_date
    available_dates = data.index.date
    valid_dates = [d for d in available_dates if d <= target_date]
    if not valid_dates:
        raise ValueError(f"No data available on or before {date_str}")
    
    closest_date = max(valid_dates)
    closest_date_idx = np.where(available_dates == closest_date)[0][0]
    
    # Get the next date index
    next_date_idx = closest_date_idx + 1
    if next_date_idx >= len(data):
        raise ValueError(f"Cannot predict for {date_str} as next trading day data is not available")
    
    next_actual_date = data.index[next_date_idx].date()
    
    # Get prices
    current_close = data['Close'].iloc[closest_date_idx]
    next_close = data['Close'].iloc[next_date_idx]
    actual_pct_change = (next_close / current_close - 1) * 100
    
    # Ensure we have all the required feature columns
    missing_cols = set(feature_cols) - set(data.columns)
    for col in missing_cols:
        data[col] = 0  # Add missing columns with zeros
    
    # Extract features in the correct order
    X = data[feature_cols].iloc[closest_date_idx-seq_length+1:closest_date_idx+1].values
    
    # Scale features
    X_scaled = scaler_X.transform(X)
    
    # Convert to tensor for model input (add batch and sequence dimensions)
    X_tensor = torch.tensor(X_scaled.reshape(1, seq_length, -1), dtype=torch.float32)
    
    # Make prediction
    model.eval()
    with torch.no_grad():
        prediction = model(X_tensor).item()
    
    # Inverse transform the prediction if needed
    if scaler_y is not None and target_type == 'pct_change':
        # For percentage targets with magnitude scaling
        prediction_sign = np.sign(prediction)
        prediction_abs = np.abs(prediction)
        prediction_abs = scaler_y.inverse_transform(np.array([[prediction_abs]]))[0][0]
        prediction = prediction_sign * prediction_abs
    
    # Calculate predicted price
    predicted_price = current_close * (1 + prediction/100)
    
    # Construct result
    result = {
        'Target Date': closest_date,
        'Target Date Close': current_close,
        'Next Trading Date': next_actual_date,
        'Next Date Close': next_close,
        'Actual % Change': actual_pct_change,
        'Predicted % Change': prediction,
        'Predicted Close': predicted_price,
        'Prediction Error': prediction - actual_pct_change,
        'Direction Correct': (prediction > 0 and actual_pct_change > 0) or (prediction < 0 and actual_pct_change < 0)
    }
    
    # Print results
    print(f"\nHistorical Prediction for {ticker}:")
    print(f"From {closest_date} to {next_actual_date}:")
    print(f"Close on {closest_date}: ${current_close:.2f}")
    print(f"Actual Close on {next_actual_date}: ${next_close:.2f}")
    print(f"Actual % Change: {actual_pct_change:.2f}%")
    print(f"Predicted % Change: {prediction:.2f}%")
    print(f"Predicted Close: ${predicted_price:.2f}")
    print(f"Prediction Error: {result['Prediction Error']:.2f}%")
    print(f"Direction Correctly Predicted: {'Yes' if result['Direction Correct'] else 'No'}")
    
    return result

def create_advanced_visualizations(predictions, actuals, dates, close_prices, ticker='UNKNOWN', threshold=0.5):
    """
    Create advanced visualizations for stock predictions including:
    - Actual vs Predicted Price Charts
    - Trading Signal Visualization (Buy/Sell)
    - Simulated Trading Performance
    
    Args:
        predictions: Model predictions (percentage changes)
        actuals: Actual percentage changes
        dates: Dates corresponding to predictions and actuals
        close_prices: Actual closing prices
        ticker: Stock ticker symbol
        threshold: Threshold for buy/sell signals (in percentage)
        
    Returns:
        Trading performance metrics
    """
    # Convert predictions and actuals to numpy arrays if they're not already
    predictions = np.array(predictions).flatten()
    actuals = np.array(actuals).flatten()
    close_prices = np.array(close_prices)
    
    # Create a DataFrame for easier manipulation
    df = pd.DataFrame({
        'Date': dates,
        'Actual_Change': actuals,
        'Predicted_Change': predictions,
        'Close': close_prices
    })
    
    # Calculate predicted prices
    df['Predicted_Price'] = df['Close'] * (1 + df['Predicted_Change'] / 100)
    
    # Calculate next day's actual price
    df['Next_Day_Price'] = df['Close'].shift(-1)
    
    # Generate trading signals (1 for buy, -1 for sell, 0 for hold)
    df['Signal'] = 0
    df.loc[df['Predicted_Change'] > threshold, 'Signal'] = 1  # Buy signal
    df.loc[df['Predicted_Change'] < -threshold, 'Signal'] = -1  # Sell signal
    
    # Simulate trading performance
    df['Position'] = df['Signal'].shift(1)  # Position taken the previous day
    df['Position'].fillna(0, inplace=True)  # No position on first day
    
    # Calculate returns
    df['Market_Return'] = df['Close'].pct_change()  # Buy and hold return
    df['Strategy_Return'] = df['Position'] * df['Market_Return']  # Strategy return
    
    # Calculate cumulative returns
    df['Cumulative_Market_Return'] = (1 + df['Market_Return']).cumprod() - 1
    df['Cumulative_Strategy_Return'] = (1 + df['Strategy_Return']).cumprod() - 1
    
    # 1. Actual vs Predicted Price Chart
    plt.figure(figsize=(16, 8))
    plt.plot(df['Date'], df['Close'], label='Actual Close Price', color='blue')
    plt.plot(df['Date'], df['Predicted_Price'], label='Predicted Close Price', color='red', alpha=0.7)
    plt.title(f'{ticker} - Actual vs Predicted Close Price', fontsize=14)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Price ($)', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(alpha=0.3)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'{ticker}_actual_vs_predicted_price.png')
    
    # 2. Trading Signal Visualization
    plt.figure(figsize=(16, 10))
    
    # Price chart with buy/sell signals
    ax1 = plt.subplot(2, 1, 1)
    plt.plot(df['Date'], df['Close'], color='blue', label='Close Price')
    
    # Plot buy signals
    buy_signals = df[df['Signal'] == 1]
    plt.scatter(buy_signals['Date'], buy_signals['Close'], marker='^', color='green', s=100, label='Buy Signal')
    
    # Plot sell signals
    sell_signals = df[df['Signal'] == -1]
    plt.scatter(sell_signals['Date'], sell_signals['Close'], marker='v', color='red', s=100, label='Sell Signal')
    
    plt.title(f'{ticker} - Trading Signals', fontsize=14)
    plt.ylabel('Price ($)', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(alpha=0.3)
    
    # Format dates
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.xticks(rotation=45)
    
    # Prediction accuracy heatmap below price chart
    ax2 = plt.subplot(2, 1, 2, sharex=ax1)
    
    # Create an array for the colors
    colors = np.zeros((len(df), 3))  # RGB array
    
    # Direction correctly predicted (green)
    correct_dir = (df['Actual_Change'] > 0) & (df['Predicted_Change'] > 0) | (df['Actual_Change'] < 0) & (df['Predicted_Change'] < 0)
    colors[correct_dir] = [0, 1, 0]  # Green
    
    # Direction incorrectly predicted (red)
    incorrect_dir = ~correct_dir
    colors[incorrect_dir] = [1, 0, 0]  # Red
    
    # Plot colored bars
    for i in range(len(df)):
        plt.axvspan(mdates.date2num(df['Date'].iloc[i]), 
                   mdates.date2num(df['Date'].iloc[i] + pd.Timedelta(days=1)), 
                   color=colors[i], alpha=0.3)
    
    # Add info to explain the colors
    legend_elements = [
        Patch(facecolor='green', alpha=0.3, label='Correct Direction'),
        Patch(facecolor='red', alpha=0.3, label='Incorrect Direction')
    ]
    plt.legend(handles=legend_elements, loc='upper right')
    
    plt.title(f'{ticker} - Prediction Direction Accuracy', fontsize=14)
    plt.ylabel('Accuracy', fontsize=12)
    plt.yticks([])  # Remove y-ticks for cleaner look
    
    plt.tight_layout()
    plt.savefig(f'{ticker}_trading_signals.png')
    
    # 3. Simulated Trading Performance
    plt.figure(figsize=(16, 8))
    plt.plot(df['Date'], (df['Cumulative_Market_Return']*100), label='Buy and Hold Strategy', color='blue')
    plt.plot(df['Date'], (df['Cumulative_Strategy_Return']*100), label='Model-based Strategy', color='green')
    plt.title(f'{ticker} - Simulated Trading Performance', fontsize=14)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Cumulative Return (%)', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(alpha=0.3)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'{ticker}_trading_performance.png')
    
    # Calculate performance metrics
    initial_capital = 10000  # $10,000 initial investment
    final_capital_market = initial_capital * (1 + df['Cumulative_Market_Return'].iloc[-1])
    final_capital_strategy = initial_capital * (1 + df['Cumulative_Strategy_Return'].iloc[-1])
    
    strategy_return = df['Cumulative_Strategy_Return'].iloc[-1] * 100
    market_return = df['Cumulative_Market_Return'].iloc[-1] * 100
    
    # Calculate more advanced metrics
    # Sharpe Ratio (assuming risk-free rate of 0% for simplicity)
    strategy_returns = df['Strategy_Return'].fillna(0)
    market_returns = df['Market_Return'].fillna(0)
    
    if len(strategy_returns) > 0 and strategy_returns.std() > 0:
        strategy_sharpe = strategy_returns.mean() / strategy_returns.std() * np.sqrt(252)  # Annualized
    else:
        strategy_sharpe = 0
        
    if len(market_returns) > 0 and market_returns.std() > 0:
        market_sharpe = market_returns.mean() / market_returns.std() * np.sqrt(252)  # Annualized
    else:
        market_sharpe = 0
    
    # Maximum Drawdown
    def calculate_max_drawdown(returns):
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative / running_max) - 1
        return drawdown.min()
    
    strategy_max_drawdown = calculate_max_drawdown(df['Strategy_Return'])
    market_max_drawdown = calculate_max_drawdown(df['Market_Return'])
    
    # Win Rate
    trades = df[df['Position'] != 0]
    if len(trades) > 0:
        winning_trades = trades[trades['Strategy_Return'] > 0]
        win_rate = len(winning_trades) / len(trades) * 100
    else:
        win_rate = 0
    
    # Print results
    print(f"\n{ticker} - TRADING PERFORMANCE SUMMARY:")
    print(f"Initial Capital: ${initial_capital:.2f}")
    print(f"Final Capital (Buy & Hold): ${final_capital_market:.2f}")
    print(f"Final Capital (Model Strategy): ${final_capital_strategy:.2f}")
    print(f"Total Return (Buy & Hold): {market_return:.2f}%")
    print(f"Total Return (Model Strategy): {strategy_return:.2f}%")
    print(f"Sharpe Ratio (Buy & Hold): {market_sharpe:.4f}")
    print(f"Sharpe Ratio (Model Strategy): {strategy_sharpe:.4f}")
    print(f"Maximum Drawdown (Buy & Hold): {market_max_drawdown*100:.2f}%")
    print(f"Maximum Drawdown (Model Strategy): {strategy_max_drawdown*100:.2f}%")
    print(f"Total Trades: {len(trades)}")
    print(f"Win Rate: {win_rate:.2f}%")
    
    performance_metrics = {
        'Initial_Capital': initial_capital,
        'Final_Capital_Market': final_capital_market,
        'Final_Capital_Strategy': final_capital_strategy,
        'Market_Return': market_return,
        'Strategy_Return': strategy_return,
        'Market_Sharpe': market_sharpe,
        'Strategy_Sharpe': strategy_sharpe,
        'Market_Max_Drawdown': market_max_drawdown,
        'Strategy_Max_Drawdown': strategy_max_drawdown,
        'Total_Trades': len(trades),
        'Win_Rate': win_rate
    }
    
    return performance_metrics, df
def simulate_investment(ticker, model, scaler_X, scaler_y, feature_cols, seq_length, 
                       investment_date, investment_amount=1000, target_type='pct_change',
                       days_to_hold=30, threshold=0.5):
    """
    Simulate investment performance starting from a specific date.
    
    Args:
        ticker: Stock ticker symbol
        model: Trained PyTorch model
        scaler_X: Feature scaler
        scaler_y: Target scaler
        feature_cols: Feature column names
        seq_length: Sequence length for the model
        investment_date: Date to start investment (format: 'YYYY-MM-DD')
        investment_amount: Initial investment amount in dollars
        target_type: Type of prediction ('pct_change', 'price', etc.)
        days_to_hold: Number of calendar days to hold the investment
        threshold: Threshold for buy/sell decisions
    
    Returns:
        Dictionary with investment performance metrics
    """
    # Parse the investment date
    start_date_obj = datetime.datetime.strptime(investment_date, '%Y-%m-%d').date()
    
    # Calculate data fetch start date (need enough data for sequence)
    fetch_start_date = start_date_obj - timedelta(days=seq_length*3 + 60)
    
    # Calculate end date
    end_date_obj = start_date_obj + timedelta(days=days_to_hold)
    
    # Convert to string format for yfinance
    fetch_start_str = fetch_start_date.strftime('%Y-%m-%d')
    end_date_str = end_date_obj.strftime('%Y-%m-%d')
    
    print(f"Simulating ${investment_amount} investment in {ticker} from {investment_date} to {end_date_str}")
    
    # Fetch data
    stock_data = yf.download(ticker, start=fetch_start_str, end=end_date_str)
    sp500_data = yf.download('^GSPC', start=fetch_start_str, end=end_date_str)
    
    # Fix multi-index columns
    stock_data.columns = stock_data.columns.get_level_values(0)
    sp500_data.columns = sp500_data.columns.get_level_values(0)
    
    # Add SP500 data
    stock_data['SP500_Close'] = sp500_data['Close']
    stock_data['SP500_Return'] = sp500_data['Close'].pct_change() * 100
    stock_data['Relative_Return'] = stock_data['Close'].pct_change() * 100 - stock_data['SP500_Return']
    stock_data['Market_Beta'] = stock_data['Close'].pct_change().rolling(60).cov(
        sp500_data['Close'].pct_change()) / sp500_data['Close'].pct_change().rolling(60).var()
    
    # Calculate technical indicators
    data = calculate_technical_indicators(stock_data)
    
    # Drop NaN values
    data.dropna(inplace=True)
    
    # Find the investment start date in the data
    # Get the closest date that's >= investment_date
    available_dates = data.index.date
    valid_dates = [d for d in available_dates if d >= start_date_obj]
    
    if not valid_dates:
        print(f"No data available on or after {investment_date}")
        return None
    
    # Get the closest date on or after the investment date
    closest_date = min(valid_dates)
    start_date_idx = np.where(available_dates == closest_date)[0][0]
    
    # Ensure we have enough data before this date for the sequence
    if start_date_idx < seq_length:
        print(f"Not enough data before {investment_date} for prediction")
        return None
    
    # Extract the investment period data
    investment_data = data.iloc[start_date_idx:]
    
    # Initial price on investment date
    initial_price = investment_data['Close'].iloc[0]
    shares_bought = investment_amount / initial_price
    
    # Make predictions for each day in the investment period
    predictions = []
    actual_changes = []
    dates = []
    
    for i in range(len(investment_data)):
        if i + start_date_idx - seq_length < 0:
            continue
            
        current_date = investment_data.index[i]
        dates.append(current_date)
        
        # Get the sequence data leading up to this day
        sequence_data = data.iloc[start_date_idx + i - seq_length:start_date_idx + i]
        
        # Extract features
        X = sequence_data[feature_cols].values
        
        # Scale features
        X_scaled = scaler_X.transform(X)
        
        # Convert to tensor
        X_tensor = torch.tensor(X_scaled.reshape(1, seq_length, -1), dtype=torch.float32)
        
        # Make prediction
        model.eval()
        with torch.no_grad():
            prediction = model(X_tensor).item()
        
        # Inverse transform if needed
        if scaler_y is not None and target_type == 'pct_change':
            prediction_sign = np.sign(prediction)
            prediction_abs = np.abs(prediction)
            prediction_abs = scaler_y.inverse_transform(np.array([[prediction_abs]]))[0][0]
            prediction = prediction_sign * prediction_abs
        
        predictions.append(prediction)
        
        # Get actual change (if available)
        if i < len(investment_data) - 1:
            today_close = investment_data['Close'].iloc[i]
            next_close = investment_data['Close'].iloc[i + 1]
            actual_change = (next_close / today_close - 1) * 100
        else:
            actual_change = 0  # No actual change for the last day
        
        actual_changes.append(actual_change)
    
    # Create DataFrame for analysis
    df = pd.DataFrame({
        'Date': dates,
        'Close': investment_data['Close'].values[:len(dates)],
        'Predicted_Change': predictions,
        'Actual_Change': actual_changes
    })
    
    # Generate trading signals
    df['Signal'] = 0
    df.loc[df['Predicted_Change'] > threshold, 'Signal'] = 1  # Buy signal
    df.loc[df['Predicted_Change'] < -threshold, 'Signal'] = -1  # Sell signal

    # Ensure predicted price is calculated if missing
    df['Predicted_Price'] = df['Close'].shift(1) * (1 + df['Predicted_Change'] / 100)    
    # Calculate daily returns
    df['Daily_Return'] = df['Close'].pct_change().fillna(0)

    
    # Simulate trading strategy
    df['Position'] = df['Signal']
    df['Position'] = df['Position'].ffill()
    
    if df['Position'].iloc[0] == 0:
        df['Position'].iloc[0] = 1  # Default to buy
    
    # Calculate strategy returns (only change position when signal changes)
    df['Strategy_Return'] = df['Daily_Return'] * df['Position']
    
    # Calculate equity curves
    df['Buy_Hold_Equity'] = investment_amount * (1 + df['Close'].pct_change().cumsum())
    df['Buy_Hold_Equity'].iloc[0] = investment_amount
    
    df['Strategy_Equity'] = investment_amount * (1 + df['Strategy_Return'].cumsum())
    df['Strategy_Equity'].iloc[0] = investment_amount
    
    # Calculate final values
    final_buy_hold = investment_amount * (df['Close'].iloc[-1] / df['Close'].iloc[0])
    final_strategy = df['Strategy_Equity'].iloc[-1]
    
    buy_hold_return = ((final_buy_hold / investment_amount) - 1) * 100
    strategy_return = ((final_strategy / investment_amount) - 1) * 100
    
    # Create visualization
    plt.figure(figsize=(16, 12))
    
    # Plot 1: Stock price and signals
    plt.subplot(3, 1, 1)
    plt.plot(df['Date'], df['Close'], label=f'{ticker} Price', color='blue')
    
    # Plot buy signals
    buy_signals = df[df['Signal'] == 1]
    plt.scatter(buy_signals['Date'], buy_signals['Close'], marker='^', color='green', s=100, label='Buy Signal')
    
    # Plot sell signals
    sell_signals = df[df['Signal'] == -1]
    plt.scatter(sell_signals['Date'], sell_signals['Close'], marker='v', color='red', s=100, label='Sell Signal')
    
    plt.title(f'{ticker} - Price and Trading Signals ({investment_date} to {end_date_str})', fontsize=14)
    plt.ylabel('Price ($)', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)
    
    # Plot 2: Predicted vs Actual changes
    plt.subplot(3, 1, 2)
    plt.plot(df['Date'], df['Predicted_Change'], label='Predicted % Change', color='orange')
    plt.plot(df['Date'], df['Actual_Change'], label='Actual % Change', color='blue', alpha=0.7)
    plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    plt.title(f'{ticker} - Predicted vs Actual % Change', fontsize=14)
    plt.ylabel('% Change', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)
    
    # Plot 3: Investment performance
    plt.subplot(3, 1, 3)
    plt.plot(df['Date'], df['Buy_Hold_Equity'], label='Buy & Hold Strategy', color='blue')
    plt.plot(df['Date'], df['Strategy_Equity'], label='Model Strategy', color='green')
    plt.title(f'{ticker} - ${investment_amount} Investment Performance', fontsize=14)
    plt.ylabel('Portfolio Value ($)', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{ticker}_investment_simulation_{investment_date.replace("-", "")}.png')
    
    # Print results
    print(f"\n{ticker} - INVESTMENT PERFORMANCE SUMMARY ({investment_date} to {end_date_str}):")
    print(f"Initial Investment: ${investment_amount:.2f}")
    print(f"Shares Purchased: {shares_bought:.4f}")
    print(f"Initial Price: ${initial_price:.2f}")
    print(f"Final Price: ${df['Close'].iloc[-1]:.2f}")
    print(f"Price Change: {((df['Close'].iloc[-1] / initial_price) - 1) * 100:.2f}%")
    print(f"Buy & Hold Strategy Final Value: ${final_buy_hold:.2f} ({buy_hold_return:.2f}%)")
    print(f"Model Strategy Final Value: ${final_strategy:.2f} ({strategy_return:.2f}%)")
    print(f"Model Outperformance: {strategy_return - buy_hold_return:.2f}%")
    
    # Calculate total trades
    position_changes = df['Position'].diff() != 0
    total_trades = position_changes.sum()
    
    print(f"Total Trades: {total_trades}")
    
    # Correct predictions count
    correct_predictions = ((df['Predicted_Change'] > 0) & (df['Actual_Change'] > 0)) | ((df['Predicted_Change'] < 0) & (df['Actual_Change'] < 0))
    direction_accuracy = correct_predictions.mean() * 100
    
    print(f"Directional Accuracy: {direction_accuracy:.2f}%")
    
    # Return metrics dictionary
    return {
        'Initial_Investment': investment_amount,
        'Shares_Purchased': shares_bought,
        'Initial_Price': initial_price,
        'Final_Price': df['Close'].iloc[-1],
        'Buy_Hold_Final': final_buy_hold,
        'Strategy_Final': final_strategy,
        'Buy_Hold_Return': buy_hold_return,
        'Strategy_Return': strategy_return,
        'Outperformance': strategy_return - buy_hold_return,
        'Total_Trades': total_trades,
        'Direction_Accuracy': direction_accuracy,
        'Data': df
    }
# Main execution
if __name__ == "__main__":
    # Configuration
    config = {
        'ticker': 'TSLA',
        'start_date': '2015-01-01',
        'seq_length': 60,  # Shorter sequence length for faster training
        'hidden_dim': 128,
        'num_layers': 2,
        'dropout': 0.3,
        'epochs': 300,
        'patience': 30,
        'target_type': 'pct_change'
    }
    
    # Prepare data
    print(f"Preparing data for {config['ticker']}...")
    (X_train, y_train, X_test, y_test, 
     train_loader, test_loader, 
     scaler_X, scaler_y, 
     test_dates, close_test, 
     original_data, feature_cols) = prepare_data(
        ticker=config['ticker'],
        start_date=config['start_date'],
        seq_length=config['seq_length'],
        target_type=config['target_type']
    )
    
    # Initialize model
    input_dim = X_train.shape[2]
    model = StockLSTM(
        input_dim=input_dim,
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
        dropout=config['dropout']
    )
    
    # Train model
    print(f"Training model...")
    model, history = train_model(
        model=model,
        train_loader=train_loader,
        X_val=X_test,
        y_val=y_test,
        num_epochs=config['epochs'],
        patience=config['patience']
    )
    
    # Evaluate model
    print(f"Evaluating model...")
    evaluation = evaluate_model(
        model=model,
        X_test=X_test,
        y_test=y_test,
        test_dates=test_dates,
        close_prices=close_test,
        original_data=original_data,
        scaler_y=scaler_y,
        ticker=config['ticker'],
        target_type=config['target_type']
    )
    
    print(f"Creating advanced trading visualizations...")
    performance_metrics, trading_df = create_advanced_visualizations(
        predictions=evaluation['Predictions'],
        actuals=evaluation['Actuals'],
        dates=test_dates,
        close_prices=close_test,
        ticker=config['ticker'],
        threshold=0.1  # Threshold for buy/sell signals (0.5% change)
    )



   # Test different thresholds to find optimal signal generation
    print(f"\nTesting different signal thresholds...")
    thresholds = [.01, .05, .1, .15, .2]  # 1%, 5%, 10%, 15%, 20%
    results = {}

    for thresh in thresholds:
        print(f"\nTesting threshold: {thresh}%")
        results[thresh], _ = create_advanced_visualizations(
            predictions=evaluation['Predictions'],
            actuals=evaluation['Actuals'],
            dates=test_dates,
            close_prices=close_test,
            ticker=f"{config['ticker']}_{thresh}",
            threshold=thresh
        )

    # Find the best performing threshold
    best_thresh = max(results, key=lambda x: results[x]['Strategy_Return'])
    print(f"\nBest performing threshold: {best_thresh}%")
    print(f"Return with best threshold: {results[best_thresh]['Strategy_Return']:.2f}%")

    # Simulate investment starting from February 17, 2025
    print(f"\nSimulating specific investment scenario...")
    investment_metrics = simulate_investment(
        ticker=config['ticker'],
        model=model,
        scaler_X=scaler_X, 
        scaler_y=scaler_y,
        feature_cols=feature_cols,
        seq_length=config['seq_length'],
        investment_date='2025-02-17',  # One month ago
        investment_amount=1000,
        target_type=config['target_type'],
        days_to_hold=30,  # Hold for 30 days
        threshold=best_thresh  # Use the best threshold from previous testing
    )
    
    create_performance_summary_chart(investment_metrics, config['ticker'])

    
    # Predict next day
    # At the end of your script, change the function call to:
    next_day_prediction, next_day_price, latest_date, next_date = predict_next_day(
        model=model,
        scaler=scaler_X,  # Changed from scaler_X to scaler
        data=original_data,  # You need to provide the data parameter
        features=feature_cols,
        sequence_length=config['seq_length'],
        target_scaler=scaler_y,  # Changed from scaler_y to target_scaler
        target_type=config['target_type']
    )
    
    # Make a prediction for a specific historical date
    print("\nTesting historical prediction capability...")
    historical_prediction = predict_for_date(
        model=model,
        scaler_X=scaler_X,
        scaler_y=scaler_y,
        feature_cols=feature_cols,
        ticker=config['ticker'],
        date_str='2023-02-21',  # Example historical date
        seq_length=config['seq_length'],
        target_type=config['target_type']
    )