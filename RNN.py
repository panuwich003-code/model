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
import pickle
import json
from skopt import gp_minimize
from sklearn.model_selection import TimeSeriesSplit

# Consider GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def save_checkpoint(checkpoint_path, trial_counter, results_list, optimizer_state=None):
    """Save checkpoint to resume later"""
    checkpoint = {
        'trial_counter': trial_counter,
        'results_list': results_list,
        'optimizer_state': optimizer_state,
        'timestamp': time.time()
    }
    
    # Save as pickle
    with open(checkpoint_path, 'wb') as f:
        pickle.dump(checkpoint, f)
    
    # Also save as JSON for human readability
    json_checkpoint = {
        'trial_counter': trial_counter,
        'results_list': results_list, # Note: results_list content must be JSON serializable
        'timestamp': checkpoint['timestamp']
    }
    json_path = checkpoint_path.replace('.pkl', '.json')
    # Helper to convert numpy types to python types for JSON
    class NpEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer): return int(obj)
            if isinstance(obj, np.floating): return float(obj)
            if isinstance(obj, np.ndarray): return obj.tolist()
            return super(NpEncoder, self).default(obj)
            
    with open(json_path, 'w') as f:
        json.dump(json_checkpoint, f, indent=2, cls=NpEncoder)
    
    print(f"Checkpoint saved at trial {trial_counter}")

def load_checkpoint(checkpoint_path):
    """Load checkpoint if exists"""
    if os.path.exists(checkpoint_path):
        try:
            with open(checkpoint_path, 'rb') as f:
                checkpoint = pickle.load(f)
            print(f"Checkpoint loaded: Trial {checkpoint['trial_counter']}")
            return checkpoint
        except Exception as e:
            print(f"Failed to load checkpoint: {e}")
            return None
    return None

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

    def fit(self, X, y, lookback=4, hidden_size=50, num_layers=2, learning_rate=0.01,
            dropout=0.1, batch_size=64, epochs=200):
        self.current_params = locals()
        del self.current_params['self']
        del self.current_params['X']
        del self.current_params['y']
        
        self.input_size = X.shape[1] if len(X.shape) > 1 else 1
        try:
            X_tensor, y_tensor = create_dataset(X, y, lookback)
            X_tensor, y_tensor = X_tensor.to(device), y_tensor.to(device)
            
            self.model = MultivariateRNN(
                self.input_size,
                hidden_size,
                num_layers=num_layers,
                dropout_rate=dropout,
                device=device
            )
            self.model = self.model.to(device)
            
            optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
            loss_fn = nn.MSELoss()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            loader = data.DataLoader(
                data.TensorDataset(X_tensor, y_tensor),
                batch_size=batch_size,
                shuffle=True,
                drop_last=True
            )
            
            for epoch in range(epochs):
                self.model.train()
                for X_batch, y_batch in loader:
                    optimizer.zero_grad()
                    y_pred = self.model(X_batch)
                    loss = loss_fn(y_pred.squeeze(), y_batch)
                    loss.backward()
                    optimizer.step()
            
            return self
        except Exception as e:
            print(f"Error in fit: {e}")
            return self

    def predict(self, X):
        if self.model is None:
            raise ValueError("Model not trained yet")
        # Ensure lookback is available
        lookback = self.current_params.get('lookback', 12) 
        
        # Create dataset for prediction (dummy y)
        X_tensor, _ = create_dataset(X, np.zeros(len(X)), lookback)
        X_tensor = X_tensor.to(device)
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(X_tensor).cpu().numpy()
        return predictions

def run_optimization(tuning_method='bayes', n_init_points=10, checkpoint_path=None, resume=True):
    results_list = []
    start_time_opt = time.time()
    trial_counter = 0
    
    # 1. Load Checkpoint
    if resume and checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = load_checkpoint(checkpoint_path)
        if checkpoint:
            trial_counter = checkpoint['trial_counter']
            results_list = checkpoint['results_list']
            print(f"Resuming... Already completed {len(results_list)} trials")
    else:
        print("Starting new optimization run")

    if tuning_method == 'bayes':
        search_space = [
            Categorical([12, 18, 24, 36, 42, 48], name='lookback'),
            Integer(8, 256, name='hidden_size'),
            Integer(1, 6, name='num_layers'),
            Real(0.001, 0.05, prior='log-uniform', name='learning_rate'),
            Integer(16, 64, name='batch_size'),
            Integer(50, 300, name='epochs')
        ]
        
        # 2. Prepare x0 (previous params) and y0 (previous scores) for Warm Start
        x0 = []
        y0 = []
        if results_list:
            print("Preparing warm start points from history...")
            for res in results_list:
                # Order must match search_space
                point = [
                    res['lookback'],
                    res['hidden_size'],
                    res['num_layers'],
                    res['learning_rate'],
                    res['batch_size'],
                    res['epochs']
                ]
                x0.append(point)
                y0.append(res['test_rmse']) # gp_minimize minimizes this value

        def objective(params):
            nonlocal trial_counter
            
            # Unpack params
            lookback, hidden_size, num_layers, learning_rate, batch_size, epochs = params
            
            # 3. Check if this set of parameters is already in history (Handling Duplicates/Warm Start)
            # This is crucial for correct resuming logic
            current_params_list = [lookback, hidden_size, num_layers, learning_rate, batch_size, epochs]
            
            # Simple check if exact params exist in x0
            if x0:
                # Compare each element because strict list equality might fail on small float diffs, 
                # but for simplicity in this context, list comparison usually works for skopt types
                if current_params_list in x0:
                    print(f"Skipping already evaluated parameters: {current_params_list}")
                    # Return the existing score
                    return y0[x0.index(current_params_list)]

            trial_counter += 1
            current_params = {
                'lookback': int(lookback),
                'hidden_size': int(hidden_size),
                'num_layers': int(num_layers),
                'learning_rate': float(learning_rate),
                'batch_size': int(batch_size),
                'epochs': int(epochs),
            }

            try:
                print(f"Trial {trial_counter} - Training with params: {current_params}")
                
                model_wrapper = RNNWrapper()
                train_pred, test_pred = [], []
                actual_train, actual_test = [], []
                train_size, test_size = 0, 0
                
                # Cross-Validation
                splits = list(tscv.split(X_scaled))
                
                for split_idx, (train_index, test_index) in enumerate(splits):
                    # print(f"Processing split {split_idx + 1}/{len(splits)}")
                    X_train, X_test = X_scaled[train_index], X_scaled[test_index]
                    y_train, y_test = y_scaled[train_index], y_scaled[test_index]
                    
                    # Reduce epochs for validation splits to speed up, full epochs for first/main
                    current_run_epochs = current_params['epochs'] if split_idx == 0 else max(1, current_params['epochs'] // 2)
                    
                    model_wrapper.fit(X_train, y_train, **{**current_params, 'epochs': current_run_epochs})
                    
                    # Store predictions
                    tr_pred = model_wrapper.predict(X_train).reshape(-1, 1)
                    te_pred = model_wrapper.predict(X_test).reshape(-1, 1)
                    
                    train_pred.append(tr_pred)
                    test_pred.append(te_pred)
                    
                    actual_train.append(y_train[current_params['lookback']:].reshape(-1, 1))
                    actual_test.append(y_test[current_params['lookback']:].reshape(-1, 1))
                    
                    train_size += len(X_train)
                    test_size += len(X_test)
                
                # Concatenate results
                train_pred = np.concatenate(train_pred)
                test_pred = np.concatenate(test_pred)
                actual_train = np.concatenate(actual_train)
                actual_test = np.concatenate(actual_test)
                
                # Inverse transform
                train_pred_actual = target_scaler.inverse_transform(train_pred)
                test_pred_actual = target_scaler.inverse_transform(test_pred)
                actual_train_inv = target_scaler.inverse_transform(actual_train)
                actual_test_inv = target_scaler.inverse_transform(actual_test)
                
                # Calculate Metrics
                train_rmse = np.sqrt(np.mean((train_pred_actual - actual_train_inv) ** 2))
                test_rmse = np.sqrt(np.mean((test_pred_actual - actual_test_inv) ** 2))
                r2 = r2_score(actual_test_inv, test_pred_actual)
                mae = mean_absolute_error(actual_test_inv, test_pred_actual)
                
                # Save results
                trial_results = {
                    **current_params,
                    'train_rmse': train_rmse,
                    'test_rmse': test_rmse,
                    'test_mae': mae,
                    'test_r2': r2,
                    'training_time': time.time() - start_time_opt,
                    'trial_number': trial_counter
                }
                results_list.append(trial_results)
                
                # Save checkpoint immediately
                if checkpoint_path:
                    save_checkpoint(checkpoint_path, trial_counter, results_list)
                
                # Save CSV log
                pd.DataFrame(results_list).to_csv(
                    f'{dir_name}/{dir_suffix}/{round_dir}/optimization_results_{tuning_method}_trial_{trial_counter}.csv',
                    index=False
                )
                
                # Plot
                plt.figure(figsize=(12, 6))
                
                # Re-construct indices for plotting (approximation for visualization)
                total_len = len(actual_train_inv) + len(actual_test_inv)
                train_indices = np.arange(len(actual_train_inv))
                test_indices = np.arange(len(actual_train_inv), total_len)
                
                plt.plot(train_indices, actual_train_inv, label='Actual Train', color='blue', alpha=0.5)
                plt.plot(train_indices, train_pred_actual, label='Train Pred', color='cyan', alpha=0.5)
                plt.plot(test_indices, actual_test_inv, label='Actual Test', color='green', alpha=0.7)
                plt.plot(test_indices, test_pred_actual, label='Test Pred', color='orange', alpha=0.7)
                
                plt.title(f"Trial {trial_counter} (RMSE: {test_rmse:.4f}) | {current_params}")
                plt.legend()
                plt.tight_layout()
                plt.savefig(f'{dir_name}/{dir_suffix}/{round_dir}/trial_{trial_counter}_results.png')
                plt.close()
                
                print(f"Trial {trial_counter} finished. Test RMSE: {test_rmse:.4f}\n")
                
                return test_rmse
                
            except Exception as e:
                print(f"Error in trial {trial_counter}: {e}")
                # Save checkpoint even on failure to avoid total loss
                if checkpoint_path:
                    save_checkpoint(checkpoint_path, trial_counter, results_list)
                return 9999.0 # Return high error on failure

        # 4. Configure Optimization Run
        print("Starting Bayesian optimization...")
        
        n_total_calls = 200
        # Calculate how many calls are left
        n_calls_remaining = n_total_calls - len(results_list)
        # Calculate how many random initial points are left
        n_init_remaining = max(0, n_init_points - len(results_list))
        
        if n_calls_remaining > 0:
            print(f"Running {n_calls_remaining} more trials (Warm start with {len(results_list)} points)")
            result = gp_minimize(
                objective,
                search_space,
                n_calls=n_calls_remaining,
                n_initial_points=n_init_remaining,
                x0=x0 if len(x0) > 0 else None, # PASS HISTORY HERE
                y0=y0 if len(y0) > 0 else None, # PASS HISTORY HERE
                random_state=42 # Set random state for reproducibility
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
            print(f"Best parameters found: {best_params}")
        else:
            print("Target number of trials already reached in checkpoint.")

    # Save final results
    final_results_df = pd.DataFrame(results_list)
    final_results_df.to_csv(f'{dir_name}/{dir_suffix}/{round_dir}/final_optimization_results_{tuning_method}.csv', index=False)
    
    if len(results_list) > 0:
        best_config = min(results_list, key=lambda x: x['test_rmse'])
        print("\nBest Configuration:")
        for key, value in best_config.items():
            print(f"{key}: {value}")
    
    return results_list

if __name__ == '__main__':
    #### Load Dataset START ####
    file_path = "d:\\Users\\Admin\\Desktop\\Data_for_PM0.1\\Official_Data_Selected.csv"
    try:
        df = pd.read_csv(file_path, encoding="utf8")
        print('original data:')
        print(df.describe().to_string(), "\n")
        
        if 'Timestamp' in df.columns:
            df['Timestamp'] = pd.to_datetime(df['Timestamp'], dayfirst=True)
            
        #### Data Preprocessing START ####
        input_features = ['Wind_Dir', 'Wind_Speed', 'Outdoor_Temperature', 'Outdoor_Humidity', 'Bar', 'Outdoor_PM2.5']
        target_feature = 'Indoor_PM0.1'
        
        df.replace(['---', ''], np.nan, inplace=True)
        
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
        
        if 'Wind_Dir' in df.columns:
            df['Wind_Dir'] = df['Wind_Dir'].apply(wind_to_degrees)
        
        for col in input_features + [target_feature]:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df = df.dropna(subset=input_features + [target_feature])
        
        df_subset = df[input_features + [target_feature]]
        z_scores = stats.zscore(df_subset)
        abs_z_scores = np.abs(z_scores)
        filtered_entries = (abs_z_scores < 3).all(axis=1)
        df = df.loc[filtered_entries.index[filtered_entries]]
        
        print('filtered data:')
        print(df.describe().to_string(), "\n")
        
        X = df[input_features].values.astype('float32')
        y = df[target_feature].values.astype('float32')
        
        input_scaler = MinMaxScaler()
        target_scaler = MinMaxScaler()
        
        X_scaled = input_scaler.fit_transform(X)
        y_scaled = target_scaler.fit_transform(y.reshape(-1, 1)).flatten()
        #### Data Preprocessing END ####
        
        tuning_method = 'bayes'
        dir_name = 'RNN_pm01_optimization'
        os.makedirs(dir_name, exist_ok=True)
        
        n_splits = [5, 10]
        n_init_points_list = [10, 20, 50]
        
        for n_split in n_splits:
            tscv = TimeSeriesSplit(n_splits=n_split)
            
            for n_init in n_init_points_list:
                dir_suffix = f"tscv_split_{n_split}_n_init_{n_init}"
                
                for round_num in range(1, 11):
                    round_dir = f"round_{round_num}"
                    os.makedirs(f'{dir_name}/{dir_suffix}/{round_dir}', exist_ok=True)
                    
                    # Create checkpoint path
                    checkpoint_path = f'{dir_name}/{dir_suffix}/{round_dir}/checkpoint.pkl'
                    
                    # Check if we should resume
                    resume = os.path.exists(checkpoint_path)
                    if resume:
                        print(f"\n{'='*60}")
                        print(f"Found checkpoint for {dir_suffix}/{round_dir}")
                        print(f"Resuming optimization...")
                        print(f"{'='*60}\n")
                    
                    # Run optimization with checkpoint support
                    results_list = run_optimization(
                        tuning_method, 
                        n_init_points=n_init,
                        checkpoint_path=checkpoint_path,
                        resume=True
                    )
                    
                    print(f"\nCompleted {dir_suffix}/{round_dir}")
                    print(f"Total trials: {len(results_list)}\n")
                    
    except FileNotFoundError:
        print(f"Error: Data file not found at {file_path}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")