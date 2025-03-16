import pandas as pd
import numpy as np
import yfinance as yf
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

class AttentionLSTM(nn.Module):
    """LSTM with self-attention mechanism for time series forecasting"""
    def __init__(self, input_size, hidden_size=128, num_layers=3, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Self-attention layer
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),  # Bidirectional: hidden_size * 2
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
            nn.Softmax(dim=1)
        )
        
        # Fully connected layers with residual connections
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.dropout1 = nn.Dropout(dropout)
        
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.bn2 = nn.BatchNorm1d(hidden_size // 2)
        self.dropout2 = nn.Dropout(dropout)
        
        self.fc3 = nn.Linear(hidden_size // 2, 1)
        
        self.leaky_relu = nn.LeakyReLU(0.1)

    def forward(self, x):
        # LSTM layers
        lstm_out, _ = self.lstm(x)  # Shape: (batch, seq_len, hidden_size*2)
        
        # Self-attention
        attention_weights = self.attention(lstm_out)  # Shape: (batch, seq_len, 1)
        context_vector = torch.sum(attention_weights * lstm_out, dim=1)  # Shape: (batch, hidden_size*2)
        
        # Fully connected layers with residual connections
        out = self.fc1(context_vector)
        out = self.bn1(out)
        out = self.leaky_relu(out)
        out = self.dropout1(out)
        
        out = self.fc2(out)
        out = self.bn2(out)
        out = self.leaky_relu(out)
        out = self.dropout2(out)
        
        return self.fc3(out)

def calculate_rsi(series, window=14):
    """Calculate Relative Strength Index"""
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def calculate_macd(series, fast=12, slow=26, signal=9):
    """Calculate Moving Average Convergence Divergence"""
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    histogram = macd - signal_line
    return macd, signal_line, histogram

def calculate_bollinger_bands(close, window=20, num_std=2):
    """Calculate Bollinger Bands"""
    sma = close.rolling(window).mean()
    std = close.rolling(window).std()
    upper = sma + (std * num_std)
    lower = sma - (std * num_std)
    return upper, sma, lower

def prepare_data(ticker='TSLA', benchmark='^GSPC', start_date='2018-01-01', end_date=None,
                seq_length=30, train_ratio=0.8):
    """Prepare data with comprehensive feature engineering"""
    
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    # Download stock data
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    benchmark_data = yf.download(benchmark, start=start_date, end=end_date)
    
    # Handle MultiIndex columns by selecting level 0
    if isinstance(stock_data.columns, pd.MultiIndex):
        # Extract only the first level of the MultiIndex
        stock_data.columns = stock_data.columns.get_level_values(0)
    
    if isinstance(benchmark_data.columns, pd.MultiIndex):
        benchmark_data.columns = benchmark_data.columns.get_level_values(0)
    
    # Basic features
    df = stock_data[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
    df['Benchmark'] = benchmark_data['Close']
    
    # Store original close prices for visualization
    close_prices = df['Close'].values
    dates = df.index
    
    # Price-based features
    df['Daily_Return'] = df['Close'].pct_change() * 100
    df['Benchmark_Return'] = df['Benchmark'].pct_change() * 100
    df['High_Low_Ratio'] = df['High'] / df['Low']
    df['Close_Open_Ratio'] = df['Close'] / df['Open']
    
    # Moving averages
    for window in [5, 10, 20, 50, 100]:
        df[f'SMA_{window}'] = df['Close'].rolling(window).mean()
        df[f'EMA_{window}'] = df['Close'].ewm(span=window, adjust=False).mean()
    
    # Moving average ratios (price momentum)
    df['Price_to_SMA_20'] = df['Close'] / df['SMA_20']
    df['Price_to_SMA_50'] = df['Close'] / df['SMA_50']
    df['Price_to_SMA_100'] = df['Close'] / df['SMA_100']
    
    # Price momentum
    for window in [5, 10, 20]:
        df[f'Momentum_{window}'] = df['Close'].pct_change(window) * 100
    
    # Volatility indicators
    for window in [5, 20, 50]:
        df[f'Volatility_{window}'] = df['Daily_Return'].rolling(window).std()
    
    # Volume indicators
    df['Volume_Change'] = df['Volume'].pct_change() * 100
    df['Volume_SMA_5'] = df['Volume'].rolling(5).mean()
    df['Volume_SMA_20'] = df['Volume'].rolling(20).mean()
    df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA_20']
    
    # Technical indicators
    df['RSI'] = calculate_rsi(df['Close'])
    macd, signal, hist = calculate_macd(df['Close'])
    df['MACD'] = macd
    df['MACD_Signal'] = signal
    df['MACD_Hist'] = hist
    
    bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(df['Close'])
    df['BB_Upper'] = bb_upper
    df['BB_Middle'] = bb_middle
    df['BB_Lower'] = bb_lower
    df['BB_Width'] = (bb_upper - bb_lower) / bb_middle
    df['BB_Position'] = (df['Close'] - bb_lower) / (bb_upper - bb_lower)
    
    # Target: Next day's percentage change
    df['Target'] = df['Close'].pct_change(1).shift(-1) * 100
    
    # Drop NaN values
    df.dropna(inplace=True)
    
    # Prepare features and target
    features = df.drop('Target', axis=1).values
    target = df['Target'].values
    feature_names = df.columns.drop('Target').tolist()
    
    # Scale features
    scaler = RobustScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Create sequences
    X, y = [], []
    for i in range(len(features_scaled) - seq_length):
        X.append(features_scaled[i:i+seq_length])
        y.append(target[i+seq_length])
    
    X, y = np.array(X), np.array(y)
    
    # Train-test split
    split_idx = int(len(X) * train_ratio)
    X_train, y_train = X[:split_idx], y[:split_idx]
    X_test, y_test = X[split_idx:], y[split_idx:]
    
    return X_train, y_train, X_test, y_test, scaler, feature_names, close_prices, dates

def train_model(X_train, y_train, X_test, y_test, input_size, hidden_size=128, 
               num_layers=3, dropout=0.3, learning_rate=0.001, batch_size=64,
               num_epochs=100, patience=15):
    """Train the LSTM model with early stopping"""
    
    # Convert to PyTorch tensors
    X_train = torch.FloatTensor(X_train)
    y_train = torch.FloatTensor(y_train.reshape(-1, 1))
    X_test = torch.FloatTensor(X_test)
    y_test = torch.FloatTensor(y_test.reshape(-1, 1))
    
    # Create data loaders
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize model
    model = AttentionLSTM(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout
    )
    
    # Loss and optimizer
    criterion = nn.HuberLoss(delta=1.0)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', patience=patience//3, factor=0.5, verbose=True
    )
    
    # Training loop with early stopping
    best_loss = float('inf')
    best_model = None
    early_stop_counter = 0
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        running_loss = 0.0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            
            # Backward pass and optimize
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            running_loss += loss.item()
        
        epoch_loss = running_loss / len(train_loader)
        train_losses.append(epoch_loss)
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_test)
            val_loss = criterion(val_outputs, y_test).item()
            val_losses.append(val_loss)
            
            # Update learning rate
            scheduler.step(val_loss)
        
        # Check for improvement
        if val_loss < best_loss:
            best_loss = val_loss
            best_model = model.state_dict().copy()
            early_stop_counter = 0
        else:
            early_stop_counter += 1
        
        # Print status
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], '
                  f'Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}')
            
            # Calculate metrics
            y_pred = val_outputs.numpy().flatten()
            y_true = y_test.numpy().flatten()
            mae = mean_absolute_error(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            
            print(f'MAE: {mae:.4f}, RMSE: {rmse:.4f}')
        
        # Early stopping
        if early_stop_counter >= patience:
            print(f'Early stopping at epoch {epoch+1}')
            break
    
    # Load best model
    model.load_state_dict(best_model)
    
    # Plot training history
    plt.figure(figsize=(12, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Loss History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('training_history.png')
    plt.close()
    
    return model, best_loss

def evaluate_model(model, X_test, y_test, close_prices=None, dates=None):
    """Evaluate model performance"""
    model.eval()
    with torch.no_grad():
        X_test_tensor = torch.FloatTensor(X_test)
        predictions = model(X_test_tensor).numpy().flatten()
    
    y_true = y_test.flatten()
    
    # Calculate metrics
    mae = mean_absolute_error(y_true, predictions)
    rmse = np.sqrt(mean_squared_error(y_true, predictions))
    
    # Calculate directional accuracy
    direction_true = np.sign(y_true)
    direction_pred = np.sign(predictions)
    directional_accuracy = np.mean(direction_true == direction_pred)
    
    print(f"Mean Absolute Error: {mae:.4f}")
    print(f"Root Mean Squared Error: {rmse:.4f}")
    print(f"Directional Accuracy: {directional_accuracy:.2%}")
    
    # Plot predictions vs actual percentage changes
    plt.figure(figsize=(14, 7))
    plt.plot(y_true[-100:], label='Actual')
    plt.plot(predictions[-100:], label='Predicted')
    plt.title('Actual vs Predicted Percentage Changes (Last 100 Days)')
    plt.xlabel('Days')
    plt.ylabel('Percentage Change')
    plt.legend()
    plt.savefig('prediction_comparison.png')
    plt.close()
    
    # Plot actual vs predicted prices if close prices are provided
    if close_prices is not None and dates is not None and len(dates) >= len(y_test):
        # Get dates for the test period
        test_dates = dates[-len(y_test):]
        
        # Calculate predicted prices using percentage changes
        base_prices = close_prices[-len(y_test)-1:-1]
        predicted_prices = base_prices * (1 + predictions/100)
        actual_prices = base_prices * (1 + y_true/100)
        
        plt.figure(figsize=(14, 7))
        plt.plot(test_dates[-100:], actual_prices[-100:], label='Actual Price', color='blue')
        plt.plot(test_dates[-100:], predicted_prices[-100:], label='Predicted Price', color='red')
        plt.title('Actual vs Predicted Stock Prices (Last 100 Days)')
        plt.xlabel('Date')
        plt.ylabel('Price ($)')
        plt.legend()
        plt.grid(True)
        plt.savefig('price_comparison.png')
        plt.close()
    
    return mae, rmse, directional_accuracy

def save_model(model, filepath='enhanced_lstm_model.pth'):
    """Save model to file"""
    torch.save({
        'model_state_dict': model.state_dict(),
        'architecture': {
            'input_size': model.lstm.input_size,
            'hidden_size': model.lstm.hidden_size,
            'num_layers': model.lstm.num_layers,
            'dropout': model.dropout1.p
        }
    }, filepath)
    print(f"Model saved to {filepath}")

def load_model(filepath='enhanced_lstm_model.pth'):
    """Load model from file"""
    checkpoint = torch.load(filepath)
    architecture = checkpoint['architecture']
    
    model = AttentionLSTM(
        input_size=architecture['input_size'],
        hidden_size=architecture['hidden_size'],
        num_layers=architecture['num_layers'],
        dropout=architecture['dropout']
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    return model

def predict_next_day(model, scaler, feature_names, ticker='TSLA', benchmark='^GSPC', seq_length=30):
    """Predict the next day's price change percentage"""
    # Get end date as today
    end_date = datetime.now().strftime('%Y-%m-%d')
    # Get start date with buffer for technical indicators
    start_date = (datetime.now() - timedelta(days=seq_length*3)).strftime('%Y-%m-%d')
    
    # Download recent data
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    benchmark_data = yf.download(benchmark, start=start_date, end=end_date)
    
    # Handle MultiIndex columns by selecting level 0
    if isinstance(stock_data.columns, pd.MultiIndex):
        stock_data.columns = stock_data.columns.get_level_values(0)
    
    if isinstance(benchmark_data.columns, pd.MultiIndex):
        benchmark_data.columns = benchmark_data.columns.get_level_values(0)
    
    if len(stock_data) < seq_length:
        print(f"Not enough data. Got {len(stock_data)} days, need {seq_length}.")
        return None
    
    # Create the same features as during training
    df = stock_data[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
    df['Benchmark'] = benchmark_data['Close']
    
    # Price-based features
    df['Daily_Return'] = df['Close'].pct_change() * 100
    df['Benchmark_Return'] = df['Benchmark'].pct_change() * 100
    df['High_Low_Ratio'] = df['High'] / df['Low']
    df['Close_Open_Ratio'] = df['Close'] / df['Open']
    
    # Moving averages
    for window in [5, 10, 20, 50, 100]:
        df[f'SMA_{window}'] = df['Close'].rolling(window).mean()
        df[f'EMA_{window}'] = df['Close'].ewm(span=window, adjust=False).mean()
    
    # Moving average ratios
    df['Price_to_SMA_20'] = df['Close'] / df['SMA_20']
    df['Price_to_SMA_50'] = df['Close'] / df['SMA_50']
    df['Price_to_SMA_100'] = df['Close'] / df['SMA_100']
    
    # Price momentum
    for window in [5, 10, 20]:
        df[f'Momentum_{window}'] = df['Close'].pct_change(window) * 100
    
    # Volatility indicators
    for window in [5, 20, 50]:
        df[f'Volatility_{window}'] = df['Daily_Return'].rolling(window).std()
    
    # Volume indicators
    df['Volume_Change'] = df['Volume'].pct_change() * 100
    df['Volume_SMA_5'] = df['Volume'].rolling(5).mean()
    df['Volume_SMA_20'] = df['Volume'].rolling(20).mean()
    df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA_20']
    
    # Technical indicators
    df['RSI'] = calculate_rsi(df['Close'])
    macd, signal, hist = calculate_macd(df['Close'])
    df['MACD'] = macd
    df['MACD_Signal'] = signal
    df['MACD_Hist'] = hist
    
    bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(df['Close'])
    df['BB_Upper'] = bb_upper
    df['BB_Middle'] = bb_middle
    df['BB_Lower'] = bb_lower
    df['BB_Width'] = (bb_upper - bb_lower) / bb_middle
    df['BB_Position'] = (df['Close'] - bb_lower) / (bb_upper - bb_lower)
    
    # Drop NaN values
    df.dropna(inplace=True)
    
    # Make sure we have enough data after feature creation
    if len(df) < seq_length:
        print(f"Not enough data after creating features. Got {len(df)}, need {seq_length}.")
        return None
    
    # Reorder columns to match training data
    if feature_names:
        # Get only the columns that exist in feature_names
        common_cols = [col for col in feature_names if col in df.columns]
        df = df[common_cols]
    
    # Get the last sequence
    last_sequence = df.values[-seq_length:]
    
    # Scale the sequence
    last_sequence_scaled = scaler.transform(last_sequence)
    
    # Convert to tensor and predict
    model.eval()
    with torch.no_grad():
        last_sequence_tensor = torch.FloatTensor(last_sequence_scaled).unsqueeze(0)
        prediction = model(last_sequence_tensor).item()
    
    return prediction

if __name__ == "__main__":
    # Configuration
    config = {
        'ticker': 'TSLA',
        'benchmark': '^GSPC',
        'start_date': '2018-01-01',
        'seq_length': 30,
        'hidden_size': 128,
        'num_layers': 3,
        'dropout': 0.4,
        'learning_rate': 0.0005,
        'batch_size': 64,
        'num_epochs': 200,
        'patience': 20
    }
    
    # Prepare data
    X_train, y_train, X_test, y_test, scaler, feature_names, close_prices, dates = prepare_data(
        ticker=config['ticker'],
        benchmark=config['benchmark'],
        start_date=config['start_date'],
        seq_length=config['seq_length']
    )
    
    # Input size is the number of features
    input_size = X_train.shape[2]
    
    # Train model
    model, best_loss = train_model(
        X_train, y_train, X_test, y_test,
        input_size=input_size,
        hidden_size=config['hidden_size'],
        num_layers=config['num_layers'],
        dropout=config['dropout'],
        learning_rate=config['learning_rate'],
        batch_size=config['batch_size'],
        num_epochs=config['num_epochs'],
        patience=config['patience']
    )
    
    # Evaluate model
    mae, rmse, dir_acc = evaluate_model(model, X_test, y_test, close_prices, dates)
    
    # Save model
    save_model(model)
    
    # Predict next day
    next_day_change = predict_next_day(
        model, scaler, feature_names,
        ticker=config['ticker'],
        benchmark=config['benchmark'],
        seq_length=config['seq_length']
    )
    
    print(f"\nPredicted price change for next trading day: {next_day_change:.2f}%")
    
    # Interpret prediction
    if next_day_change > 1.5:
        print("Recommendation: Strong Buy")
    elif next_day_change > 0.5:
        print("Recommendation: Buy")
    elif next_day_change < -1.5:
        print("Recommendation: Strong Sell")
    elif next_day_change < -0.5:
        print("Recommendation: Sell")
    else:
        print("Recommendation: Hold")