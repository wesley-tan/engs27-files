import numpy as np
import pandas as pd
from typing import Tuple, List, Dict
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import math
from scipy.integrate import odeint
from scipy.optimize import minimize
from numpy.random import normal

class ImprovedSEIRDCovidSimulator:
    def __init__(self, data_path: str, country: str, start_date: str, end_date: str):
        """
        Initialize improved SEIRD simulator with time-varying parameters and vaccination effects.
        """
        self.data = pd.read_csv(data_path, parse_dates=['Date'])
        self.country = country
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.training_end_date = self.start_date + timedelta(days=60)
        
        # Initialize variant parameters
        self.variant_effects = {
            'transmission_boost': 1.0,  # Will be updated based on date
            'severity_boost': 1.0
        }
        
        # Load and process data
        self._process_country_data()
        
        # Calculate parameters using improved method
        self.calculate_model_parameters()
        
        # Initialize states
        self.initialize_states()

    def _process_country_data(self):
        """Enhanced data processing with improved cleaning, validation, and normalization."""
        print(f"\n[INFO] Processing data for {self.country}")
        print(f"[INFO] Date range: {self.start_date} to {self.end_date}")
        
        # Get country data and aggregate by date
        self.country_data = self.data[
            (self.data['Country/Region'] == self.country) &
            (self.data['Date'] >= self.start_date) &
            (self.data['Date'] <= self.end_date)
        ].groupby('Date')[['Confirmed', 'Deaths', 'Recovered']].sum().reset_index()
        
        # Smooth the data using 7-day rolling average
        for col in ['Confirmed', 'Deaths', 'Recovered']:
            self.country_data[f'{col}_Raw'] = self.country_data[col]
            self.country_data[col] = self.country_data[col].rolling(7, min_periods=1).mean()
        
        # Calculate active cases with improved methodology
        self.country_data['Active_Raw'] = (
            self.country_data['Confirmed'] -
            self.country_data['Deaths'] -
            self.country_data['Recovered']
        ).clip(0)
        
        # Apply 7-day rolling average with centered window
        self.country_data['Active'] = (
            self.country_data['Active_Raw']
            .rolling(7, center=True, min_periods=1)
            .mean()
        )
        
        # Calculate daily new active cases
        self.country_data['New_Active'] = (
            self.country_data['Active'].diff()
            .fillna(0)
            .clip(0)
        )
        
        # Adjust active cases based on recovery time
        recovery_period = 14  # typical COVID-19 recovery period
        self.country_data['Active_Adjusted'] = (
            self.country_data['New_Active']
            .rolling(window=recovery_period, min_periods=1)
            .sum()
        ).clip(0)
        
        # Use the adjusted active cases for training/testing
        self.country_data['Active'] = self.country_data['Active_Adjusted']
        
        # Calculate daily changes
        for col in ['Confirmed', 'Deaths', 'Recovered']:
            self.country_data[f'New_{col}'] = self.country_data[col].diff().fillna(0).clip(0)
        
        # Split data
        self.training_data = self.country_data[
            self.country_data['Date'] <= self.training_end_date
        ].copy()
        
        self.testing_data = self.country_data[
            self.country_data['Date'] > self.training_end_date
        ].copy()
        
        # Estimate true population affected using multiple methods
        reported_cases = self.country_data['Confirmed'].max()
        estimated_multiplier = self._estimate_underreporting()
        self.total_population = reported_cases * estimated_multiplier
        
        # Normalize the data columns
        columns_to_normalize = ['Confirmed', 'Deaths', 'Recovered', 'Active']
        
        # Store the scaling factors for later denormalization if needed
        self.scaling_factors = {}
        
        for col in columns_to_normalize:
            max_val = self.country_data[col].max()
            if max_val > 0:
                self.scaling_factors[col] = max_val
                self.country_data[f'{col}_Normalized'] = self.country_data[col] / max_val
            else:
                self.scaling_factors[col] = 1
                self.country_data[f'{col}_Normalized'] = self.country_data[col]
        
        # Also normalize the training and testing splits
        for col in columns_to_normalize:
            self.training_data[f'{col}_Normalized'] = (
                self.training_data[col] / self.scaling_factors[col]
            )
            self.testing_data[f'{col}_Normalized'] = (
                self.testing_data[col] / self.scaling_factors[col]
            )
        
        print("\n[INFO] Data Normalization Factors:")
        for col, factor in self.scaling_factors.items():
            print(f"{col}: {factor:,.2f}")
        
        print(f"[INFO] Training data size: {len(self.training_data)} days")
        print(f"[INFO] Testing data size: {len(self.testing_data)} days")
        print(f"[INFO] Estimated total population affected: {self.total_population:,.0f}")

    def _estimate_underreporting(self) -> float:
        """Estimate underreporting factor using multiple indicators."""
        # Method 1: CFR-based estimation
        observed_cfr = (self.training_data['Deaths'].max() / 
                       self.training_data['Confirmed'].max())
        expected_cfr = 0.02  # Expected CFR based on global data
        cfr_multiplier = min(observed_cfr / expected_cfr, 10) if observed_cfr > 0 else 2
        
        # Method 2: Testing rate-based estimation (simplified)
        testing_multiplier = 2.5
        
        # Combine estimates with weights
        final_multiplier = (0.7 * cfr_multiplier + 0.3 * testing_multiplier)
        return max(2, min(final_multiplier, 5))  # Constrain between 2x and 5x

    def calculate_model_parameters(self):
        """Enhanced parameter estimation using optimization and Monte Carlo for uncertainties."""
        print("\n[INFO] Starting parameter optimization...")
        self.params = self._optimize_parameters()
        print("[INFO] Base parameters calculated:", self.params)
        
        # Initialize time-varying parameters
        self.beta_base = self.params['beta']
        self.sigma = self.params['sigma']
        self.gamma_base = self.params['gamma']
        self.mu_base = self.params['mu']
        
        print("[INFO] Starting Monte Carlo sampling for uncertainty estimation...")
        
        # Calculate uncertainties using Monte Carlo sampling
        n_samples = 1000  # Increased number of samples for better statistics
        param_variations = {
            'beta': [],
            'gamma': [],
            'mu': []
        }
        
        # Perform Monte Carlo sampling
        for _ in range(n_samples):
            # Sample parameters with relative uncertainties
            param_variations['beta'].append(
                normal(self.beta_base, 0.1 * self.beta_base)  # 10% variation
            )
            param_variations['gamma'].append(
                normal(self.gamma_base, 0.1 * self.gamma_base)
            )
            param_variations['mu'].append(
                normal(self.mu_base, 0.1 * self.mu_base)
            )
        
        # Calculate uncertainties as standard deviations
        self.param_uncertainties = {
            'beta': np.std(param_variations['beta']),
            'gamma': np.std(param_variations['gamma']),
            'mu': np.std(param_variations['mu'])
        }
        
        # Add confidence intervals
        confidence_intervals = {
            param: np.percentile(variations, [2.5, 97.5])
            for param, variations in param_variations.items()
        }
        
        print("\n[DEBUG] Optimized Parameters with 95% Confidence Intervals:")
        for param, value in self.params.items():
            if param in self.param_uncertainties:
                ci = confidence_intervals[param]
                print(f"{param}: {value:.4f} ± {self.param_uncertainties[param]:.4f}")
                print(f"    95% CI: [{ci[0]:.4f}, {ci[1]:.4f}]")
            else:
                print(f"{param}: {value:.4f}")

    def _optimize_parameters(self) -> Dict[str, float]:
        """Optimize model parameters using scipy.optimize."""
        print("[INFO] Starting parameter optimization...")
        
        def objective(params):
            beta, sigma, gamma, mu = params
            results = self._run_basic_seird(beta, sigma, gamma, mu)
            
            active_error = np.mean((results['I'] - 
                                  self.training_data['Active'].values / self.total_population) ** 2)
            death_error = np.mean((results['D'] - 
                                 self.training_data['Deaths'].values / self.total_population) ** 2)
            
            total_error = active_error + 2 * death_error
            print(f"[DEBUG] Current optimization error: {total_error:.6f}")
            return total_error
        
        # Initial parameter guess
        initial_guess = [0.3, 1/5.2, 0.1, 0.01]
        
        # Parameter bounds
        bounds = [(0.1, 0.5), (1/7, 1/4), (0.05, 0.2), (0.001, 0.05)]
        
        # Optimize
        result = minimize(objective, initial_guess, bounds=bounds, method='L-BFGS-B')
        
        print("[INFO] Parameter optimization complete")
        return {
            'beta': result.x[0],
            'sigma': result.x[1],
            'gamma': result.x[2],
            'mu': result.x[3]
        }

    def _run_basic_seird(self, beta, sigma, gamma, mu):
        """Basic SEIRD model for parameter optimization."""
        def seird_derivatives(state, t, beta, sigma, gamma, mu):
            S, E, I, R, D = state
            dSdt = -beta * S * I
            dEdt = beta * S * I - sigma * E
            dIdt = sigma * E - (gamma + mu) * I
            dRdt = gamma * I
            dDdt = mu * I
            return [dSdt, dEdt, dIdt, dRdt, dDdt]
        
        # Initial conditions from first day of training
        I0 = self.training_data['Active'].iloc[0] / self.total_population
        R0 = self.training_data['Recovered'].iloc[0] / self.total_population
        D0 = self.training_data['Deaths'].iloc[0] / self.total_population
        E0 = I0 * 0.5
        S0 = 1 - (I0 + R0 + D0 + E0)
        
        # Time points
        t = np.arange(len(self.training_data))
        
        # Solve ODE
        solution = odeint(seird_derivatives, [S0, E0, I0, R0, D0], t, 
                         args=(beta, sigma, gamma, mu))
        
        return {
            'S': solution[:, 0],
            'E': solution[:, 1],
            'I': solution[:, 2],
            'R': solution[:, 3],
            'D': solution[:, 4]
        }

    def initialize_states(self):
        """Initialize model states using exact values from end of training period."""
        # Get the last day of training data
        last_training_day = self.training_data.iloc[-1]
        
        # Calculate exact proportions from last training day
        total_population = self.country_data['Confirmed'].max()
        
        # Set initial states to match exactly with training data end point
        self.I = last_training_day['Active'] / total_population
        self.R = last_training_day['Recovered'] / total_population
        self.D = last_training_day['Deaths'] / total_population
        
        # Estimate exposed based on new cases trend
        new_cases_rate = (last_training_day['Active'] - self.training_data['Active'].iloc[-2]) / total_population
        self.E = max(0, new_cases_rate * 5)  # Assume ~5 days incubation period worth of exposed
        
        # Calculate susceptible as remainder to ensure total = 1
        self.S = 1 - (self.I + self.R + self.D + self.E)
        
        # Debug output
        print("\n[DEBUG] Initial State Values:")
        print(f"Susceptible: {self.S:.4f}")
        print(f"Exposed: {self.E:.4f}")
        print(f"Infected: {self.I:.4f}")
        print(f"Recovered: {self.R:.4f}")
        print(f"Deceased: {self.D:.4f}")
        print(f"Total: {self.S + self.E + self.I + self.R + self.D:.4f}")

    def _get_time_varying_parameters(self, day: int):
        """Calculate time-varying parameters based on various factors."""
        # Base seasonal modification
        seasonal_mod = self.get_seasonal_modifier(day)
        
        # Intervention effect (simplified example)
        intervention_effect = max(0.6, 1 - (day / 365))  # Gradually improving interventions
        
        # Variant effect (simplified example)
        if day > 100:  # Assume variant emergence after 100 days
            self.variant_effects['transmission_boost'] = min(2.0, 1.0 + (day - 100) / 200)
            self.variant_effects['severity_boost'] = min(1.5, 1.0 + (day - 100) / 400)
        
        # Combine effects
        effective_beta = (self.beta_base * seasonal_mod * intervention_effect * 
                         self.variant_effects['transmission_boost'])
        effective_gamma = self.gamma_base * intervention_effect
        effective_mu = self.mu_base * self.variant_effects['severity_boost']
        
        return effective_beta, effective_gamma, effective_mu

    def step(self, day: int):
        """Step function using transition matrix after training period."""
        # Get time-varying parameters
        beta, gamma, mu = self._get_time_varying_parameters(day)
        
        # Get current state vector
        current_state = np.array([self.S, self.E, self.I, self.R, self.D])
        
        # Get transition matrix
        transition_matrix = get_seird_transition_matrix(
            beta, self.sigma, gamma, mu, self.S, self.I
        ).values
        
        # Update states using matrix multiplication
        new_state = np.dot(current_state, transition_matrix)
        
        # Update instance variables
        self.S, self.E, self.I, self.R, self.D = new_state
        
        # Ensure non-negative values and conservation of population
        total = sum(new_state)
        self.S = max(0, self.S / total)
        self.E = max(0, self.E / total)
        self.I = max(0, self.I / total)
        self.R = max(0, self.R / total)
        self.D = max(0, self.D / total)
        
        return {
            'Susceptible': self.S,
            'Exposed': self.E,
            'Infected': self.I,
            'Recovered': self.R,
            'Deceased': self.D,
            'Parameters': {
                'beta': beta,
                'gamma': gamma,
                'mu': mu
            }
        }

    def get_seasonal_modifier(self, day: int):
        """Enhanced seasonal modification with multiple cycles."""
        annual_cycle = 1 + 0.2 * math.sin(2 * math.pi * day / 365)
        weekly_cycle = 1 + 0.05 * math.sin(2 * math.pi * day / 7)
        return annual_cycle * weekly_cycle

    def plot_prediction_comparison(self, sim_results: pd.DataFrame, save_plots: bool = True):
        """Plot prediction results against real data."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
        
        # Get proper normalization base using peak active cases
        peak_active = self.country_data['Active'].max()
        normalization_base = max(peak_active * 1.5, self.total_population * 0.01)
        
        # Training period
        training_days = np.arange(len(self.training_data))
        
        # Testing period
        n_pred_points = len(self.testing_data)
        test_days = np.arange(len(self.training_data), len(self.training_data) + n_pred_points)
        
        # Get the last point of training data for proper scaling
        last_training_active = self.training_data['Active'].iloc[-1] / normalization_base
        last_training_recovered = self.training_data['Recovered'].iloc[-1] / normalization_base
        
        # Scale simulation results to match the last training point
        active_scale = last_training_active / sim_results['Infected'].iloc[0]
        recovered_scale = last_training_recovered / sim_results['Recovered'].iloc[0]
        
        # Plot Active Cases
        ax1.plot(training_days, 
                self.training_data['Active'] / normalization_base,
                'b-', label='Training Data (Active)', linewidth=2)
        ax1.plot(test_days,
                self.testing_data['Active'] / normalization_base,
                'g--', label='Real Data (Active)', linewidth=2)
        ax1.plot(test_days,
                sim_results['Infected'].values[:n_pred_points] * active_scale,
                'r-', label='Predicted Active', linewidth=2)
        
        # Plot Recovered Cases
        ax2.plot(training_days,
                self.training_data['Recovered'] / normalization_base,
                'b-', label='Training Data (Recovered)', linewidth=2)
        ax2.plot(test_days,
                self.testing_data['Recovered'] / normalization_base,
                'g--', label='Real Data (Recovered)', linewidth=2)
        ax2.plot(test_days,
                sim_results['Recovered'].values[:n_pred_points] * recovered_scale,
                'r-', label='Predicted Recovered', linewidth=2)
        
        # Add vertical line to mark training/prediction split
        for ax in [ax1, ax2]:
            ax.axvline(x=len(self.training_data), color='k', linestyle='--', alpha=0.5)
            ax.set_xlim(0, len(self.country_data))
            ax.grid(True, alpha=0.3)
            ax.legend()
        
        ax1.set_title('Active Cases Comparison')
        ax2.set_title('Recovered Cases Comparison')
        ax2.set_xlabel('Days from Start Date')
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig(f'{self.country}_prediction_comparison.png', dpi=300, bbox_inches='tight')
        
        plt.show()

    def calculate_prediction_error(self, sim_results: pd.DataFrame) -> Dict[str, float]:
        """Calculate error metrics with detailed debugging."""
        # Get proper normalization base using peak active cases
        peak_active = self.country_data['Active'].max()
        normalization_base = max(peak_active * 1.5, self.total_population * 0.01)
        
        # Ensure arrays are same length
        n_points = min(len(self.testing_data), len(sim_results))
        
        # Get real data
        real_active = self.testing_data['Active'].values[:n_points] / normalization_base
        real_recovered = self.testing_data['Recovered'].values[:n_points] / normalization_base
        
        # Get predicted data
        pred_active = sim_results['Infected'].values[:n_points] * (peak_active / normalization_base)
        pred_recovered = sim_results['Recovered'].values[:n_points] * (peak_active / normalization_base)
        
        # Print debug information
        print("\n[DEBUG] Data Ranges:")
        print(f"Real Active: min={real_active.min():.4f}, max={real_active.max():.4f}")
        print(f"Pred Active: min={pred_active.min():.4f}, max={pred_active.max():.4f}")
        print(f"Real Recovered: min={real_recovered.min():.4f}, max={real_recovered.max():.4f}")
        print(f"Pred Recovered: min={pred_recovered.min():.4f}, max={pred_recovered.max():.4f}")
        
        def calculate_metrics(real, pred, name):
            # Remove any NaN or infinite values
            mask = np.isfinite(real) & np.isfinite(pred)
            real = real[mask]
            pred = pred[mask]
            
            if len(real) == 0:
                return 0, 0, 0, 0
                
            mse = np.mean((real - pred) ** 2)
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(real - pred))
            
            # Calculate R² with more information
            ss_res = np.sum((real - pred) ** 2)
            ss_tot = np.sum((real - np.mean(real)) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
            
            print(f"\n[DEBUG] {name} Metrics Details:")
            print(f"Sum of Squared Residuals: {ss_res:.4f}")
            print(f"Total Sum of Squares: {ss_tot:.4f}")
            print(f"Variance in real data: {np.var(real):.4f}")
            print(f"Mean absolute error: {mae:.4f}")
            
            return mse, rmse, mae, r2
        
        # Calculate metrics with debugging info
        active_metrics = calculate_metrics(real_active, pred_active, "Active")
        recovered_metrics = calculate_metrics(real_recovered, pred_recovered, "Recovered")
        
        return {
            'Active_MSE': active_metrics[0],
            'Active_RMSE': active_metrics[1],
            'Active_MAE': active_metrics[2],
            'Active_R2': active_metrics[3],
            'Recovered_MSE': recovered_metrics[0],
            'Recovered_RMSE': recovered_metrics[1],
            'Recovered_MAE': recovered_metrics[2],
            'Recovered_R2': recovered_metrics[3]
        }

def get_seird_transition_matrix(beta, sigma, gamma, mu, S, I):
    """
    Create the normalized SEIRD transition matrix with self-transitions
    States order: [S, E, I, R, D]
    
    Parameters:
    - beta: transmission rate
    - sigma: progression rate from exposed to infected
    - gamma: recovery rate
    - mu: mortality rate
    - S: current susceptible population (normalized)
    - I: current infected population (normalized)
    """
    
    # Calculate self-transition probabilities
    p_s_to_e = beta * I
    p_s_to_s = 1 - p_s_to_e
    
    p_e_to_i = sigma
    p_e_to_e = 1 - p_e_to_i
    
    p_i_to_r = gamma
    p_i_to_d = mu
    p_i_to_i = 1 - (p_i_to_r + p_i_to_d)
    
    # Create transition matrix with self-transitions
    transition_matrix = np.array([
        [p_s_to_s, p_s_to_e,        0,        0,        0],    # From S
        [       0,  p_e_to_e,  p_e_to_i,      0,        0],    # From E
        [       0,         0,  p_i_to_i, p_i_to_r, p_i_to_d],  # From I
        [       0,         0,         0,        1,        0],    # From R (absorbing)
        [       0,         0,         0,        0,        1]     # From D (absorbing)
    ])
    
    states = ['S', 'E', 'I', 'R', 'D']
    transition_df = pd.DataFrame(
        transition_matrix,
        columns=[f'To_{s}' for s in states],
        index=[f'From_{s}' for s in states]
    )
    
    return transition_df

# Example usage with some typical values:
beta = 0.3    # transmission rate
sigma = 1/5.2  # progression rate (≈ 5.2 days incubation period)
gamma = 0.1    # recovery rate
mu = 0.01     # mortality rate
S = 0.9       # example susceptible population
I = 0.1       # example infected population

print("SEIRD Transition Matrix:")
print(get_seird_transition_matrix(beta, sigma, gamma, mu, S, I))

if __name__ == "__main__":
    # Create simulator instance
    simulator = ImprovedSEIRDCovidSimulator(
        data_path="covid_19_clean_complete.csv",
        country="Cambodia",
        start_date="2020-01-22",
        end_date="2020-05-22"
    )

    # Run simulation for 100 days
    print("\n[INFO] Running simulation...")
    results = []
    for day in range(100):
        if day % 10 == 0:
            print(f"[INFO] Simulating day {day}")
        step_result = simulator.step(day)
        results.append(step_result)

    # Convert results to DataFrame
    sim_results = pd.DataFrame(results)

    # Plot the predictions comparison
    simulator.plot_prediction_comparison(sim_results)

    # Calculate and display error metrics
    error_metrics = simulator.calculate_prediction_error(sim_results)

    # Print all metrics
    print("\nError Metrics:")
    print("Active Cases:")
    print(f"MSE: {error_metrics['Active_MSE']:.6f}")
    print(f"RMSE: {error_metrics['Active_RMSE']:.6f}")
    print(f"MAE: {error_metrics['Active_MAE']:.6f}")
    print(f"R²: {error_metrics['Active_R2']:.6f}")

    print("\nRecovered Cases:")
    print(f"MSE: {error_metrics['Recovered_MSE']:.6f}")
    print(f"RMSE: {error_metrics['Recovered_RMSE']:.6f}")
    print(f"MAE: {error_metrics['Recovered_MAE']:.6f}")
    print(f"R²: {error_metrics['Recovered_R2']:.6f}")
