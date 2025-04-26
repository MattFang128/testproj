import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.covariance import EmpiricalCovariance

class EigenAdjustedCovariance:
    """
    A class for eigen-adjusting covariance matrices with automatic calibration
    based on simulated bias estimates.
    """
    
    def __init__(self, adjustment_method="auto_calibrated"):
        """
        Initialize the eigen-adjustment model.
        
        Parameters:
        -----------
        adjustment_method : str
            Method to use for adjustment: "auto_calibrated", "linear_shrinkage", 
            "clipping", "nonlinear_shrinkage"
        """
        self.adjustment_method = adjustment_method
        self.bias_stats = None
        self.calibrated_params = {}
        
    def estimate_bias(self, n_assets, n_obs, n_simulations=1000, factor_structure=True):
        """
        Estimate eigenvalue biases through simulation.
        
        Parameters:
        -----------
        n_assets : int
            Number of assets
        n_obs : int
            Number of observations
        n_simulations : int
            Number of simulation runs
        factor_structure : bool
            If True, use a factor model for the true covariance
            
        Returns:
        --------
        self
        """
        print(f"Estimating eigenvalue biases with {n_simulations} simulations...")
        
        # Storage for bias results
        eigenval_bias_ratios = np.zeros((n_simulations, n_assets))
        
        # Create a "true" covariance matrix
        if factor_structure:
            # Factor model with exponentially decaying eigenvalues
            true_eigenvals = 10 * np.exp(-np.arange(n_assets) / (n_assets/5))
            random_eigenvecs = linalg.qr(np.random.randn(n_assets, n_assets))[0]
            true_cov = random_eigenvecs @ np.diag(true_eigenvals) @ random_eigenvecs.T
        else:
            # Simple diagonal structure with different variances
            true_eigenvals = np.linspace(1, 10, n_assets)
            true_cov = np.diag(true_eigenvals)
        
        for i in range(n_simulations):
            # Generate sample data from true covariance
            data = np.random.multivariate_normal(np.zeros(n_assets), true_cov, size=n_obs)
            
            # Compute sample covariance
            sample_cov = np.cov(data, rowvar=False)
            
            # Extract eigenvalues and sort them
            sample_eigenvals = np.sort(linalg.eigvalsh(sample_cov))
            
            # Store bias ratios for each eigenvalue
            eigenval_bias_ratios[i] = sample_eigenvals / true_eigenvals
        
        # Compute statistics on bias
        self.bias_stats = {
            'mean_bias_ratio': np.mean(eigenval_bias_ratios, axis=0),
            'median_bias_ratio': np.median(eigenval_bias_ratios, axis=0),
            'std_bias_ratio': np.std(eigenval_bias_ratios, axis=0),
            'true_eigenvals': true_eigenvals,
            'n_assets': n_assets,
            'n_obs': n_obs,
            'ratio_n_t': n_assets / n_obs
        }
        
        # Calibrate adjustment parameters based on bias statistics
        self._calibrate_parameters()
        
        return self
    
    def _calibrate_parameters(self):
        """Calibrate adjustment parameters based on bias statistics."""
        if self.bias_stats is None:
            raise ValueError("Must estimate bias before calibrating parameters")
        
        # Extract bias statistics
        mean_bias = self.bias_stats['mean_bias_ratio']
        n_assets = self.bias_stats['n_assets']
        ratio_n_t = self.bias_stats['ratio_n_t']
        
        # Calibrate linear shrinkage intensity based on average bias
        # Higher bias = more shrinkage needed
        avg_deviation = np.mean(np.abs(mean_bias - 1))
        alpha = min(0.9, max(0.1, avg_deviation))
        self.calibrated_params['linear_shrinkage_alpha'] = alpha
        
        # Calibrate minimum eigenvalue threshold based on bias in smallest eigenvalues
        # Set threshold higher when small eigenvalues are more underestimated
        smallest_bias = mean_bias[0]  # Bias of smallest eigenvalue
        min_eigenval = max(1e-6, (1 - smallest_bias) * 0.1) if smallest_bias < 1 else 1e-6
        self.calibrated_params['min_eigenval'] = min_eigenval
        
        # Calibrate nonlinear shrinkage parameters based on distribution of bias
        # More extreme bias distribution = more nonlinear transformation needed
        max_bias = np.max(mean_bias)
        min_bias = np.min(mean_bias)
        bias_range = max_bias - min_bias
        self.calibrated_params['nonlinear_power'] = 0.5 * (1 + min(0.5, ratio_n_t))
        
        print("Calibrated parameters based on bias estimates:")
        for param, value in self.calibrated_params.items():
            print(f"  {param}: {value:.6f}")
    
    def fit(self, returns=None, sample_cov=None):
        """
        Fit the model on return data or a sample covariance matrix.
        
        Parameters:
        -----------
        returns : ndarray, optional
            Asset returns matrix (observations x assets)
        sample_cov : ndarray, optional
            Sample covariance matrix (must provide either returns or sample_cov)
            
        Returns:
        --------
        self
        """
        if returns is None and sample_cov is None:
            raise ValueError("Must provide either returns or sample_cov")
        
        if sample_cov is None:
            # Calculate sample covariance from returns
            self.sample_cov = np.cov(returns, rowvar=False)
        else:
            self.sample_cov = sample_cov
        
        # Get dimensions
        n_assets = self.sample_cov.shape[0]
        
        # If bias not estimated and using auto calibration, estimate with current dimensions
        if self.bias_stats is None and self.adjustment_method == "auto_calibrated":
            if returns is not None:
                n_obs = returns.shape[0]
                self.estimate_bias(n_assets=n_assets, n_obs=n_obs)
            else:
                # Assume reasonable observation count if not provided
                n_obs = max(100, n_assets * 2)
                self.estimate_bias(n_assets=n_assets, n_obs=n_obs)
        
        # Adjust the covariance matrix
        self.adjusted_cov = self._adjust_covariance()
        
        return self
    
    def _adjust_covariance(self):
        """Adjust covariance matrix using the specified method."""
        # Ensure the matrix is symmetric (handle numerical issues)
        sample_cov = (self.sample_cov + self.sample_cov.T) / 2
        
        # Perform eigendecomposition
        eigenvals, eigenvecs = linalg.eigh(sample_cov)
        
        # Handle negative eigenvalues first (if any)
        eigenvals = np.maximum(eigenvals, 0)
        
        if self.adjustment_method == "auto_calibrated":
            # Use bias statistics to determine the best adjustment
            if self.bias_stats is None:
                raise ValueError("Bias statistics must be estimated for auto_calibrated method")
            
            # Apply combined adjustment based on calibrated parameters
            mean_eigenval = np.mean(eigenvals)
            
            # 1. Apply linear shrinkage based on calibrated alpha
            alpha = self.calibrated_params['linear_shrinkage_alpha']
            shrunk_eigenvals = (1 - alpha) * eigenvals + alpha * mean_eigenval
            
            # 2. Apply minimum eigenvalue threshold
            min_eigenval = self.calibrated_params['min_eigenval']
            clipped_eigenvals = np.maximum(shrunk_eigenvals, min_eigenval)
            
            # 3. Apply nonlinear transformation for the largest eigenvalues
            power = self.calibrated_params['nonlinear_power']
            max_eigenval = np.max(clipped_eigenvals)
            indices_large = clipped_eigenvals > mean_eigenval
            
            # Only transform large eigenvalues
            adjusted_eigenvals = clipped_eigenvals.copy()
            if any(indices_large):
                adjusted_eigenvals[indices_large] = mean_eigenval + (
                    (clipped_eigenvals[indices_large] - mean_eigenval) ** power
                ) * (max_eigenval - mean_eigenval) ** (1 - power)
            
        elif self.adjustment_method == "linear_shrinkage":
            # Simple linear shrinkage to the mean
            alpha = self.calibrated_params.get('linear_shrinkage_alpha', 0.5)
            mean_eigenval = np.mean(eigenvals)
            adjusted_eigenvals = (1 - alpha) * eigenvals + alpha * mean_eigenval
            
        elif self.adjustment_method == "clipping":
            # Clip small eigenvalues to ensure minimum value
            min_eigenval = self.calibrated_params.get('min_eigenval', 1e-6)
            adjusted_eigenvals = np.maximum(eigenvals, min_eigenval)
            
        elif self.adjustment_method == "nonlinear_shrinkage":
            # Nonlinear shrinkage
            power = self.calibrated_params.get('nonlinear_power', 0.5)
            mean_eigenval = np.mean(eigenvals)
            adjusted_eigenvals = mean_eigenval + (eigenvals - mean_eigenval) ** power
            
        else:
            raise ValueError(f"Unknown method: {self.adjustment_method}")
        
        # Reconstruct the adjusted covariance matrix
        adjusted_cov = eigenvecs @ np.diag(adjusted_eigenvals) @ eigenvecs.T
        
        return adjusted_cov
    
    def plot_bias(self):
        """Plot eigenvalue bias statistics."""
        if self.bias_stats is None:
            raise ValueError("Must estimate bias before plotting")
        
        plt.figure(figsize=(12, 8))
        
        # Plot mean bias ratio
        mean_bias = self.bias_stats['mean_bias_ratio']
        std_bias = self.bias_stats['std_bias_ratio']
        true_eigenvals = self.bias_stats['true_eigenvals']
        
        eigenval_indices = np.arange(len(true_eigenvals))
        
        plt.subplot(2, 1, 1)
        plt.errorbar(eigenval_indices, mean_bias, yerr=std_bias, fmt='o-', 
                     capsize=5, label='Mean Bias ± Std Dev')
        plt.axhline(y=1.0, color='r', linestyle='--', label='No Bias')
        plt.xlabel('Eigenvalue Index (smallest to largest)')
        plt.ylabel('Bias Ratio (Sample/True)')
        plt.title('Eigenvalue Bias by Position')
        plt.legend()
        plt.grid(True)
        
        # Plot true vs. expected sample eigenvalues
        plt.subplot(2, 1, 2)
        plt.plot(eigenval_indices, true_eigenvals, 'o-', label='True Eigenvalues')
        plt.plot(eigenval_indices, true_eigenvals * mean_bias, 'x-', 
                 label='Expected Sample Eigenvalues')
        plt.xlabel('Eigenvalue Index (smallest to largest)')
        plt.ylabel('Eigenvalue Magnitude')
        plt.title('True vs. Expected Sample Eigenvalues')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()
        
    def plot_adjustments(self):
        """Plot original vs. adjusted eigenvalues."""
        if not hasattr(self, 'adjusted_cov'):
            raise ValueError("Must fit the model before plotting adjustments")
        
        # Get eigenvalues
        orig_eigenvals = linalg.eigvalsh(self.sample_cov)
        adj_eigenvals = linalg.eigvalsh(self.adjusted_cov)
        
        # Sort eigenvalues in ascending order
        orig_eigenvals = np.sort(orig_eigenvals)
        adj_eigenvals = np.sort(adj_eigenvals)
        
        # Create indices
        indices = np.arange(len(orig_eigenvals))
        
        plt.figure(figsize=(12, 6))
        plt.plot(indices, orig_eigenvals, 'o-', label='Original Eigenvalues')
        plt.plot(indices, adj_eigenvals, 'x-', label='Adjusted Eigenvalues')
        if self.bias_stats is not None:
            if len(self.bias_stats['true_eigenvals']) == len(orig_eigenvals):
                plt.plot(indices, self.bias_stats['true_eigenvals'], 's-', 
                         label='True Eigenvalues (from simulation)')
        
        plt.xlabel('Eigenvalue Index (smallest to largest)')
        plt.ylabel('Eigenvalue Magnitude')
        plt.title(f'Original vs. Adjusted Eigenvalues ({self.adjustment_method})')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

def run_portfolio_backtest(returns, methods=None, window_size=252, min_history=126):
    """
    Run a portfolio backtest using different covariance matrix adjustment methods.
    
    Parameters:
    -----------
    returns : DataFrame
        Asset returns DataFrame (time x assets)
    methods : list, optional
        List of adjustment methods to evaluate
    window_size : int
        Rolling window size for covariance estimation
    min_history : int
        Minimum history required before constructing portfolios
        
    Returns:
    --------
    DataFrame
        Portfolio returns for each method
    """
    if methods is None:
        methods = ["unadjusted", "auto_calibrated", "linear_shrinkage", "clipping"]
    
    T, N = returns.shape
    portfolio_returns = pd.DataFrame(index=returns.index)
    
    print(f"Running portfolio backtest with {T} periods and {N} assets...")
    
    for t in range(min_history, T):
        if t % 50 == 0:
            print(f"Processing period {t} of {T}...")
            
        # Get historical data window
        history_end = t
        history_start = max(0, history_end - window_size)
        history = returns.iloc[history_start:history_end]
        
        # For each method, calculate optimal weights
        for method in methods:
            # Get covariance matrix
            if method == "unadjusted":
                cov_matrix = history.cov().values
            else:
                # Use the EigenAdjustedCovariance class
                model = EigenAdjustedCovariance(adjustment_method=method)
                model.fit(returns=history.values)
                cov_matrix = model.adjusted_cov
            
            try:
                # Calculate minimum variance portfolio weights
                inv_cov = linalg.inv(cov_matrix)
                ones = np.ones(N)
                weights = inv_cov @ ones
                weights = weights / np.sum(weights)
                
                # Calculate portfolio return for the next period
                next_return = returns.iloc[t].values @ weights
                portfolio_returns.loc[returns.index[t], method] = next_return
                
            except np.linalg.LinAlgError:
                print(f"Warning: Singular matrix in period {t} for method {method}")
                portfolio_returns.loc[returns.index[t], method] = np.nan
    
    return portfolio_returns

def evaluate_portfolio_performance(portfolio_returns):
    """
    Evaluate and visualize backtest performance.
    
    Parameters:
    -----------
    portfolio_returns : DataFrame
        Portfolio returns for each method
        
    Returns:
    --------
    DataFrame
        Performance metrics
    """
    # Calculate cumulative returns
    cumulative_returns = (1 + portfolio_returns).cumprod()
    
    # Plot cumulative returns
    plt.figure(figsize=(12, 6))
    cumulative_returns.plot()
    plt.title('Cumulative Portfolio Returns')
    plt.ylabel('Cumulative Return')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # Calculate performance metrics
    annual_factor = 252  # Assuming daily returns
    
    metrics = pd.DataFrame(index=portfolio_returns.columns)
    
    # Annualized return
    metrics['Annual Return'] = portfolio_returns.mean() * annual_factor
    
    # Annualized volatility
    metrics['Annual Volatility'] = portfolio_returns.std() * np.sqrt(annual_factor)
    
    # Sharpe ratio (assuming 0 risk-free rate for simplicity)
    metrics['Sharpe Ratio'] = metrics['Annual Return'] / metrics['Annual Volatility']
    
    # Maximum drawdown
    max_drawdown = []
    for col in portfolio_returns.columns:
        cumul = cumulative_returns[col]
        running_max = cumul.cummax()
        drawdown = (cumul / running_max - 1)
        max_drawdown.append(drawdown.min())
    
    metrics['Max Drawdown'] = max_drawdown
    
    # Calmar ratio
    metrics['Calmar Ratio'] = metrics['Annual Return'] / np.abs(metrics['Max Drawdown'])
    
    # Print metrics
    print("\nPortfolio Performance Metrics:")
    print(metrics)
    
    return metrics

# Example usage
if __name__ == "__main__":
    print("End-to-End Eigen-Adjustment with Bias Calibration Example")
    
    # Simulate returns data for demonstration
    np.random.seed(42)
    n_assets = 30
    n_days = 1000
    
    print(f"Generating synthetic return data ({n_days} days, {n_assets} assets)...")
    
    # Create asset names
    asset_names = [f"Asset_{i+1}" for i in range(n_assets)]
    
    # Generate dates
    dates = pd.date_range(end=pd.Timestamp.today(), periods=n_days)
    
    # Generate factor structure
    n_factors = 3
    factor_exposures = np.random.randn(n_assets, n_factors)
    factor_returns = np.random.randn(n_days, n_factors) * 0.01
    specific_returns = np.random.randn(n_days, n_assets) * 0.02
    returns_data = specific_returns + factor_returns @ factor_exposures.T
    
    # Convert to dataframe
    returns_df = pd.DataFrame(returns_data, index=dates, columns=asset_names)
    
    # 1. Create the model and estimate biases
    model = EigenAdjustedCovariance(adjustment_method="auto_calibrated")
    model.estimate_bias(n_assets=n_assets, n_obs=252, n_simulations=500)
    
    # 2. Visualize the bias
    model.plot_bias()
    
    # 3. Fit on the full dataset (for demonstration)
    print("\nFitting the model on the full dataset...")
    model.fit(returns=returns_df.values)
    
    # 4. Visualize the adjustments
    model.plot_adjustments()
    
    # 5. Run portfolio backtest
    print("\nRunning portfolio backtest...")
    portfolio_returns = run_portfolio_backtest(
        returns=returns_df, 
        methods=["unadjusted", "auto_calibrated", "linear_shrinkage", "clipping"],
        window_size=252,
        min_history=252
    )
    
    # 6. Evaluate performance
    metrics = evaluate_portfolio_performance(portfolio_returns)
    
    print("\nDone!")
