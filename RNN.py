import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from scipy import stats
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from skopt.space import Real, Integer, Categorical 
import time
import os
from skopt import gp_minimize
from sklearn.model_selection import TimeSeriesSplit

# Consider GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def create_dataset(X, y, lookback):
    if len(X) <= lookback:
        raise ValueError("Lookback period longer than dataset")
    X_list, y_list = [], []
    for i in range(len(X) - lookback):
        X_list.append(X[i:i + lookback])
        y_list.append(y[i + lookback])
    X_array = np.array(X_list)
    y_array = np.array(y_list)
    return torch.from_numpy(X_array).float(), torch.from_numpy(y_array).float()

class MultivariateRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout_rate, device):
        super().__init__()
        self.device = device
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            device=device
        ).to(device)
        self.linear = nn.Linear(hidden_size, 1).to(device)

    def forward(self, x):
        rnn_out, _ = self.rnn(x)
        last_time_step = rnn_out[:, -1, :]
        predictions = self.linear(last_time_step)
        return predictions

class RNNWrapper:
    def __init__(self):
        self.model = None
        self.input_size = None
        self.current_params = None

    def get_params(self, deep=True):
        return {}

    def set_params(self, **params):
        return self

    def fit(self, X, y, lookback=4, hidden_size=50, num_layers=2, learning_rate=0.01,
            dropout=0.1, batch_size=64, epochs=200):
        start_time = time.time()
        self.current_params = locals()
        del self.current_params['self']
        del self.current_params['X']
        del self.current_params['y']
        
        # Set input size based on the data
        self.input_size = X.shape[1] if len(X.shape) > 1 else 1
        try:
            # Create datasets
            X_tensor, y_tensor = create_dataset(X, y, lookback)
            X_tensor, y_tensor = X_tensor.to(device), y_tensor.to(device)
            
            # Initialize model
            self.model = MultivariateRNN(
                self.input_size,
                hidden_size,
                num_layers=num_layers,
                dropout_rate=dropout,
                device=device
            )
            self.model = self.model.to(device)
            
            # Training setup
            optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
            loss_fn = nn.MSELoss()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            loader = data.DataLoader(
                data.TensorDataset(X_tensor, y_tensor),
                batch_size=batch_size,
                shuffle=True,
                drop_last=True  # Ensure all batches are the same size

            )
            
            # Training loop
            for epoch in range(epochs):
                self.model.train()
                epoch_losses = []
                for X_batch, y_batch in loader:
                    optimizer.zero_grad()
                    y_pred = self.model(X_batch)
                    loss = loss_fn(y_pred.squeeze(), y_batch)
                    loss.backward()
                    optimizer.step()
                    epoch_losses.append(loss.item())
                
                if epoch % 50 == 0:
                    print(f"Epoch {epoch}/{epochs}, Loss: {np.mean(epoch_losses):.4f}")
            
            return self
        except Exception as e:
            print(f"Error in fit: {e}")
            return self

    def predict(self, X):
        if self.model is None:
            raise ValueError("Model not trained yet")
        X_tensor, _ = create_dataset(X, np.zeros(len(X)), self.current_params['lookback'])
        X_tensor = X_tensor.to(device)
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(X_tensor).cpu().numpy()
        return predictions

    def score(self, X, y):
        """Return negative MSE score for optimization"""
        try:
            predictions = self.predict(X)
            y_seq = y[self.current_params['lookback']:]
            mse = np.mean((predictions.squeeze() - y_seq) ** 2)
            return -mse
        except Exception as e:
            print(f"Error in score: {e}")
            return float('-inf')

def run_optimization(tuning_method='bayes', n_init_points=10):
    results_list = []
    start_time = time.time()
    trial_counter = 0

    if tuning_method == 'bayes':
        search_space = [
            Categorical([12, 18, 24, 36, 42, 48], name='lookback'),
            Integer(8, 256, name='hidden_size'),
            Integer(1, 6, name='num_layers'),
            Real(0.001, 0.05, prior='log-uniform', name='learning_rate'),
            Integer(16, 64, name='batch_size'),
            Integer(50, 300, name='epochs')
        ]
        # param_space = {
        #     'sequence_length': Categorical([12, 18, 24, 36 ,42 ,48 ]),
        #     'hidden_size': Integer(8, 512),
        #     'num_layers': Integer(1, 10),
        #     'learning_rate': Real(0.0001, 0.001, prior='log-uniform'),
        #     'batch_size': Categorical([16, 32, 64,]),
        #     'n_epochs': Integer(50, 1000)
        #         }
        def objective(params):
            nonlocal trial_counter
            trial_counter += 1
            lookback, hidden_size, num_layers, learning_rate, batch_size, epochs = params
            
            current_params = {
                'lookback': (lookback),
                'hidden_size': int(hidden_size),
                'num_layers': int(num_layers),
                'learning_rate': float(learning_rate),
                'batch_size': int(batch_size),
                'epochs': int(epochs),
            }

            try:
                print(f"Trial {trial_counter} - Training with params: {current_params}")
                
                # Train model with current parameters
                model = RNNWrapper()
                train_size, test_size = 0, 0
                actual_train, actual_test = [], []
                train_pred, test_pred = [], []
                
                # Get split indices
                splits = list(tscv.split(X_scaled))
                
                for split_idx, (train_index, test_index) in enumerate(splits):
                    print(f"\nProcessing split {split_idx + 1}/{len(splits)}")
                    X_train, X_test = X_scaled[train_index], X_scaled[test_index]
                    y_train, y_test = y_scaled[train_index], y_scaled[test_index]
                    
                    # If not first split, reduce epochs to avoid overtraining
                    current_epochs = current_params['epochs'] if split_idx == 0 else current_params['epochs'] // 2
                    
                    # Fit model with current cumulative data
                    model.fit(X_train, y_train,
                              **{**current_params, 'epochs': current_epochs})
                    
                    # Get predictions and reshape them to 2D arrays
                    train_predictions = model.predict(X_train).reshape(-1, 1)
                    test_predictions = model.predict(X_test).reshape(-1, 1)
                    train_pred.append(train_predictions)
                    test_pred.append(test_predictions)
                    
                    # Reshape actual values to 2D arrays
                    actual_train.append(y_train[current_params['lookback']:].reshape(-1, 1))
                    actual_test.append(y_test[current_params['lookback']:].reshape(-1, 1))
                    
                    train_size += len(X_train)
                    test_size += len(X_test)
                
                # Concatenate predictions and actual values
                train_pred = np.concatenate(train_pred)
                test_pred = np.concatenate(test_pred)
                actual_train = np.concatenate(actual_train)
                actual_test = np.concatenate(actual_test)
                
                # Inverse transform to get actual values
                train_pred_actual = target_scaler.inverse_transform(train_pred)
                test_pred_actual = target_scaler.inverse_transform(test_pred)
                actual_train = target_scaler.inverse_transform(actual_train)
                actual_test = target_scaler.inverse_transform(actual_test)
                
                train_rmse = np.sqrt(np.mean((train_pred_actual - actual_train) ** 2))
                test_rmse = np.sqrt(np.mean((test_pred_actual - actual_test) ** 2))
                r2 = r2_score(actual_test, test_pred_actual)
                mae = mean_absolute_error(actual_test, test_pred_actual)
                
                # Save results for this trial
                trial_results = {
                    **current_params,
                    'train_rmse': train_rmse,
                    'test_rmse': test_rmse,
                    'test_mae': mae,
                    'test_r2': r2,
                    'training_time': time.time() - start_time,
                    'trial_number': trial_counter
                }
                results_list.append(trial_results)
                
                # Save trial results to CSV after each trial
                pd.DataFrame(results_list).to_csv(
                    f'{dir_name}/{dir_suffix}/{round_dir}/optimization_results_{tuning_method}_trial_{trial_counter}.csv',
                    index=False
                )
                
                # Plot results
                plt.figure(figsize=(16, 10))
                plt.suptitle(f"Trial {trial_counter} - Multivariate RNN Prediction")
                
                train_indices = np.arange(current_params['lookback'], len(actual_train) + current_params['lookback'])
                test_indices = np.arange(len(actual_train) + current_params['lookback'],
                                         len(actual_train) + len(actual_test) + current_params['lookback'])
                
                plt.plot(train_indices, actual_train, label='Actual Train', color='blue', alpha=0.7)
                plt.plot(train_indices, train_pred_actual, label='Train Predictions', color='red', alpha=0.7)
                plt.plot(test_indices, actual_test, label='Actual Test', color='green', alpha=0.7)
                plt.plot(test_indices, test_pred_actual, label='Test Predictions', color='orange', alpha=0.7)
                plt.axvline(x=train_size, color='purple', linestyle='--', label='Train/Test Split')
                
                plt.title(f"Trial {trial_counter} Predictions (Test RMSE: {test_rmse:.4f})")
                plt.xlabel("Time Index")
                plt.ylabel("PM 0.1")
                plt.legend()
                plt.grid(True)
                plt.tight_layout()
                plt.savefig(f'{dir_name}/{dir_suffix}/{round_dir}/trial_{trial_counter}_results.png')
                plt.close()
                
                print(f"Trial {trial_counter} completed in {trial_results['training_time']:.2f} seconds with test RMSE: {test_rmse:.4f} \n\n")
                
                return test_rmse
            except Exception as e:
                print(f"Error in trial {trial_counter}: {e}")
                return float('inf')

        # Run Bayesian optimization
        print("Starting Bayesian optimization...")
        result = gp_minimize(
            objective,
            search_space,
            n_calls=200,
            n_initial_points=n_init_points,
        )
        print("Bayesian optimization completed.")
        
        best_params = {
            'lookback': int(result.x[0]),
            'hidden_size': int(result.x[1]),
            'num_layers': int(result.x[2]),
            'learning_rate': float(result.x[3]),
            'batch_size': int(result.x[4]),
            'epochs': int(result.x[5])
        }
        
        results_list.append({
            **best_params,
            'train_rmse': None,
            'test_rmse': result.fun,
            'training_time': time.time() - start_time,
            'trial_number': trial_counter
        })
        
        print(f"Best parameters found: {best_params}")

    # Save final combined results
    final_results_df = pd.DataFrame(results_list)
    final_results_df.to_csv(f'{dir_name}/{dir_suffix}/{round_dir}/final_optimization_results_{tuning_method}.csv', index=False)
    
    # Print best performing configuration
    if len(results_list) > 0:
        best_config = min(results_list, key=lambda x: x['test_rmse'])
        print("\nBest Configuration:")
        for key, value in best_config.items():
            print(f"{key}: {value}")
    
    return results_list

if __name__ == '__main__':
    #### Load Dataset START ####
    file_path = "d:\\Users\\Admin\\Desktop\\Data_for_PM0.1\\Official_Data_Selected.csv"
    df = pd.read_csv(file_path, encoding="utf8")
    print('original data:')
    print(df.describe().to_string(), "\n")
    
    # Set index to datetime using column 'Timestamp'
    if 'Timestamp' in df.columns:
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], dayfirst=True)
    #### Load Dataset END ####
    
    #### Data Preprocessing START ####
    # Select features
    input_features = ['Wind_Dir', 'Wind_Speed', 'Outdoor_Temperature', 'Outdoor_Humidity', 'Bar', 'Outdoor_PM2.5']
    target_feature = 'Indoor_PM0.1'
    
    # Clean '---' or empty strings to NaN first
    df.replace(['---', ''], np.nan, inplace=True)
    
    # Function to convert Wind Direction to Degrees
    def wind_to_degrees(direction):
        directions = {
            "N": 0, "NNE": 22.5, "NE": 45, "ENE": 67.5,
            "E": 90, "ESE": 112.5, "SE": 135, "SSE": 157.5,
            "S": 180, "SSW": 202.5, "SW": 225, "WSW": 247.5,
            "W": 270, "WNW": 292.5, "NW": 315, "NNW": 337.5
        }
        if isinstance(direction, str):
            return directions.get(direction.upper(), np.nan)
        return direction
    
    # Apply conversion if column exists
    if 'Wind_Dir' in df.columns:
        df['Wind_Dir'] = df['Wind_Dir'].apply(wind_to_degrees)
    
    # Convert all input columns to numeric, coercing errors to NaN
    for col in input_features + [target_feature]:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Drop NaNs BEFORE calculating Z-score
    df = df.dropna(subset=input_features + [target_feature])
    
    # Remove outliers
    df_subset = df[input_features + [target_feature]]
    z_scores = stats.zscore(df_subset)
    abs_z_scores = np.abs(z_scores)
    filtered_entries = (abs_z_scores < 3).all(axis=1)
    df = df.loc[filtered_entries.index[filtered_entries]]
    
    # Data Summary
    print('filtered data:')
    print(df.describe().to_string(), "\n")
    
    # Prepare the dataset
    X = df[input_features].values.astype('float32')
    y = df[target_feature].values.astype('float32')
    
    # Separate scalers for input features and target
    input_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()
    
    X_scaled = input_scaler.fit_transform(X)
    y_scaled = target_scaler.fit_transform(y.reshape(-1, 1)).flatten()
    #### Data Preprocessing END ####
    
    # Run optimization with chosen method
    tuning_method = 'bayes'
    
    # Set the directory and round number for this optimization
    dir_name = 'RNN_pm01_optimization'
    os.makedirs(dir_name, exist_ok=True)
    
    # Set the time series split set [5, 10]
    n_splits = [5, 10]
    n_init_points = [10, 20, 50]
    
    for n_split in n_splits:
        # Time series split
        tscv = TimeSeriesSplit(n_splits=n_split)
        
        for n_init in n_init_points:
            # Create a unique directory name
            dir_suffix = f"tscv_split_{n_split}_n_init_{n_init}"
            
            for round_num in range(1, 11):
                round_dir = f"round_{round_num}"
                os.makedirs(f'{dir_name}/{dir_suffix}/{round_dir}', exist_ok=True)
                
                results_list = run_optimization(tuning_method, n_init_points=n_init)
                
                # Save results
                results_df = pd.DataFrame(results_list)
                results_df.to_csv(f'{dir_name}/{dir_suffix}/{round_dir}/optimization_results_{tuning_method}.csv', index=False)