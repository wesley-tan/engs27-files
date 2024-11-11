import numpy as np
import pandas as pd
from typing import Tuple, List, Dict
import random
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from datetime import datetime, timedelta
import math


class CovidPredictionSimulator:
    # Grid state constants
    EMPTY = 0
    SUSCEPTIBLE = 1
    INFECTED_NON_QUARANTINED = 2
    INFECTED_QUARANTINED = 3
    RECOVERED_NON_QUARANTINED = 4
    RECOVERED_QUARANTINED = 5

    def __init__(self, data_path: str, country: str, start_date: str, end_date: str):
        """
        Initialize simulator using first 60 days for training and remaining for prediction.

        Parameters:
        -----------
        data_path: Path to COVID-19 data CSV
        country: Country to analyze
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        """
        self.data = pd.read_csv(data_path, parse_dates=['Date'])
        self.country = country
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)

        # Set training period (first 60 days)
        self.training_end_date = self.start_date + timedelta(days=60)

        # Load and process data
        self._process_country_data()

        # Calculate parameters from training data
        self.calculate_daily_parameters()

        # Initialize simulation
        self.initialize_simulation_params()

    def _process_country_data(self):
        """Process and validate country-specific data."""
        # Get country data and aggregate by date
        self.country_data = self.data[
            (self.data['Country/Region'] == self.country) &
            (self.data['Date'] >= self.start_date) &
            (self.data['Date'] <= self.end_date)
            ].groupby('Date')[['Confirmed', 'Deaths', 'Recovered']].sum().reset_index()

        # Calculate active cases
        self.country_data['Active'] = (
            self.country_data['Confirmed'] -
            self.country_data['Deaths'] -
            self.country_data['Recovered']
        ).clip(0)  # Ensure no negative active cases

        # Add debug printing to verify data
        print("\n[DEBUG] Data Verification:")
        print(f"First day active cases: {self.country_data['Active'].iloc[0]}")
        print(f"Last day active cases: {self.country_data['Active'].iloc[-1]}")
        print(f"Max active cases: {self.country_data['Active'].max()}")
        print(f"Number of unique active case values: {len(self.country_data['Active'].unique())}")

        # Calculate new cases using difference
        self.country_data['New_Cases'] = self.country_data['Confirmed'].diff().fillna(0)

        # Split into training and testing periods
        self.training_data = self.country_data[
            self.country_data['Date'] <= self.training_end_date
            ].copy()

        self.testing_data = self.country_data[
            self.country_data['Date'] > self.training_end_date
            ].copy()

        # Verify testing data
        print("\n[DEBUG] Testing Data Verification:")
        print(f"Testing period start: {self.testing_data['Date'].iloc[0]}")
        print(f"Testing period end: {self.testing_data['Date'].iloc[-1]}")
        print(f"Number of testing days: {len(self.testing_data)}")
        print(f"Testing data active cases range: {self.testing_data['Active'].min()} to {self.testing_data['Active'].max()}")

        # Calculate proportions for proper scaling
        total_population = self.country_data['Confirmed'].max()
        self.training_data['Active_Ratio'] = self.training_data['Active'] / total_population
        self.training_data['Recovered_Ratio'] = self.training_data['Recovered'] / total_population
        self.testing_data['Active_Ratio'] = self.testing_data['Active'] / total_population
        self.testing_data['Recovered_Ratio'] = self.testing_data['Recovered'] / total_population

    def calculate_daily_parameters(self):
        """Calculate transmission and recovery parameters using refined approach."""
        window = 14  # Use 14-day window for better trend capture
        
        # Calculate active cases and susceptible population
        active_cases = self.training_data['Active'].replace(0, 1)
        total_population = self.country_data['Confirmed'].max()
        susceptible = total_population - self.training_data['Confirmed']
        susceptible = susceptible.replace(0, 1)
        
        # Calculate effective reproduction number (R0) from training data
        growth_rate = (self.training_data['New_Cases'] / active_cases).rolling(window=window).mean()
        effective_r0 = growth_rate.mean() * 14  # Multiply by average infectious period
        
        # Set beta based on R0 and current population state
        base_transmission = effective_r0 / 14  # Daily transmission rate
        
        # Set more conservative transmission rates
        self.beta_n = min(max(0.1, base_transmission), 0.3)
        self.beta_q = self.beta_n * 0.25  # Quarantined transmission is 25% of non-quarantined
        
        # Calculate recovery rates from actual data
        recovery_rate = (
            self.training_data['Recovered'].diff() / 
            active_cases
        ).rolling(window=window).mean()
        
        # Set recovery rates based on observed data with realistic bounds
        median_recovery = recovery_rate.median()
        self.gamma_n = max(0.05, min(0.12, median_recovery))  # Non-quarantined recovery
        self.gamma_q = self.gamma_n * 1.3  # Quarantined recovery is 30% faster
        
        # Calculate quarantine ratio based on healthcare capacity
        self.quarantine_ratio = max(0.3, min(0.6, (
            self.training_data['Recovered'] / 
            self.training_data['Confirmed'].replace(0, 1)
        ).mean()))
        
        print("\n[DEBUG] Calibrated Parameters:")
        print(f"Effective R0: {effective_r0:.4f}")
        print(f"Beta (Non-quarantined): {self.beta_n:.4f}")
        print(f"Beta (Quarantined): {self.beta_q:.4f}")
        print(f"Gamma (Non-quarantined): {self.gamma_n:.4f}")
        print(f"Gamma (Quarantined): {self.gamma_q:.4f}")
        print(f"Quarantine Ratio: {self.quarantine_ratio:.4f}")
        
        # Additional debugging information
        print("\n[DEBUG] Training Data Statistics:")
        print(f"Average daily growth rate: {growth_rate.mean():.4f}")
        print(f"Average daily recovery rate: {recovery_rate.mean():.4f}")
        print(f"Peak active cases: {self.training_data['Active'].max() / total_population:.4f}")

    def initialize_simulation_params(self):
        """Initialize simulation using exact values from end of training period."""
        # Calculate parameters from first two months
        self.calculate_daily_parameters()

        # Grid parameters
        self.grid_size = 100
        self.empty_ratio = 0.2

        # Get exact state at end of training period
        last_training_day = self.training_data.iloc[-1]
        total_population = self.country_data['Confirmed'].max()

        # Calculate exact proportions from last training day
        self.initial_infected_ratio = last_training_day['Active'] / total_population
        self.initial_recovered_ratio = last_training_day['Recovered'] / total_population

        # Adjust the infected ratio to match expected values
        active_ratio = self.initial_infected_ratio
        recovered_ratio = self.initial_recovered_ratio

        print("\n[DEBUG] Initial State:")
        print(f"Expected Active: {self.initial_infected_ratio:.4f}")
        print(f"Expected Recovered: {self.initial_recovered_ratio:.4f}")

        # Mobility rates based on transmission patterns
        base_mobility = 5.0
        self.mobility = {
            self.SUSCEPTIBLE: base_mobility,
            self.INFECTED_NON_QUARANTINED: base_mobility * 0.8,
            self.INFECTED_QUARANTINED: 0.0,
            self.RECOVERED_NON_QUARANTINED: base_mobility,
            self.RECOVERED_QUARANTINED: 0.0
        }

        self.initialize_prediction_grid(active_ratio=active_ratio, recovered_ratio=recovered_ratio)

    def initialize_prediction_grid(self, active_ratio, recovered_ratio):
        """Initialize grid with exact ratios for active, recovered, and quarantined cases."""
        max_retries = 10  # Add maximum retry limit
        best_grid = None
        best_diff = float('inf')
        
        for attempt in range(max_retries):
            self.grid = np.zeros((self.grid_size, self.grid_size), dtype=int)
            total_cells = self.grid_size * self.grid_size

            # Calculate exact number of cells needed for each state
            empty_cells = int(total_cells * self.empty_ratio)
            remaining_cells = total_cells - empty_cells

            # Use ceil and floor to ensure exact numbers
            initial_infected_cells = int(np.ceil(remaining_cells * active_ratio))
            initial_recovered_cells = int(np.floor(remaining_cells * recovered_ratio))
            initial_susceptible = remaining_cells - initial_infected_cells - initial_recovered_cells

            # Create list of all positions and shuffle
            all_positions = list(range(total_cells))
            random.shuffle(all_positions)
            current_pos = 0

            # Fill the grid deterministically
            # Empty cells
            for _ in range(empty_cells):
                i, j = all_positions[current_pos] // self.grid_size, all_positions[current_pos] % self.grid_size
                self.grid[i, j] = self.EMPTY
                current_pos += 1

            # Infected cells
            quarantined_infected = int(initial_infected_cells * self.quarantine_ratio)
            non_quarantined_infected = initial_infected_cells - quarantined_infected
            
            for _ in range(quarantined_infected):
                i, j = all_positions[current_pos] // self.grid_size, all_positions[current_pos] % self.grid_size
                self.grid[i, j] = self.INFECTED_QUARANTINED
                current_pos += 1
                
            for _ in range(non_quarantined_infected):
                i, j = all_positions[current_pos] // self.grid_size, all_positions[current_pos] % self.grid_size
                self.grid[i, j] = self.INFECTED_NON_QUARANTINED
                current_pos += 1

            # Recovered cells
            quarantined_recovered = int(initial_recovered_cells * self.quarantine_ratio)
            non_quarantined_recovered = initial_recovered_cells - quarantined_recovered
            
            for _ in range(quarantined_recovered):
                i, j = all_positions[current_pos] // self.grid_size, all_positions[current_pos] % self.grid_size
                self.grid[i, j] = self.RECOVERED_QUARANTINED
                current_pos += 1
                
            for _ in range(non_quarantined_recovered):
                i, j = all_positions[current_pos] // self.grid_size, all_positions[current_pos] % self.grid_size
                self.grid[i, j] = self.RECOVERED_NON_QUARANTINED
                current_pos += 1

            # Fill remaining with susceptible
            while current_pos < total_cells:
                i, j = all_positions[current_pos] // self.grid_size, all_positions[current_pos] % self.grid_size
                self.grid[i, j] = self.SUSCEPTIBLE
                current_pos += 1

            # Check accuracy
            stats = self.get_statistics()
            active_ratio_sim = stats['Active_NonQuarantined'] + stats['Active_Quarantined']
            recovered_ratio_sim = stats['Recovered_NonQuarantined'] + stats['Recovered_Quarantined']
            
            current_diff = abs(active_ratio_sim - active_ratio) + abs(recovered_ratio_sim - recovered_ratio)
            
            if current_diff < best_diff:
                best_diff = current_diff
                best_grid = self.grid.copy()
            
            if current_diff < 0.01:  # 1% tolerance
                break
                
        # Use best found grid if no perfect match
        if best_grid is not None:
            self.grid = best_grid
        
        # Final stats
        stats = self.get_statistics()
        print("\nInitialized Grid State (Final):")
        print(f"Active Ratio: {stats['Active_NonQuarantined'] + stats['Active_Quarantined']:.4f} (Target: {active_ratio:.4f})")
        print(f"Recovered Ratio: {stats['Recovered_NonQuarantined'] + stats['Recovered_Quarantined']:.4f} (Target: {recovered_ratio:.4f})")

    def get_neighbors(self, i: int, j: int) -> List[Tuple[int, int]]:
        """Get valid neighboring cells."""
        neighbors = []
        for di, dj in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            ni, nj = (i + di) % self.grid_size, (j + dj) % self.grid_size
            neighbors.append((ni, nj))
        return neighbors

    def random_walk(self, day: int):
        """Enhanced random walk with realistic movement patterns."""
        # Dynamic social distancing based on active cases
        current_active = (np.sum(self.grid == self.INFECTED_NON_QUARANTINED) + 
                         np.sum(self.grid == self.INFECTED_QUARANTINED)) / (self.grid_size * self.grid_size)
        
        # Stronger distancing when more cases are active
        social_distance_factor = max(0.2, min(1.0, 1 - (current_active * 3)))
        
        new_grid = self.grid.copy()
        moved = set()  # Track moved cells to avoid double movement
        
        # Process cells in random order
        indices = [(i, j) for i in range(self.grid_size) for j in range(self.grid_size)]
        random.shuffle(indices)
        
        for i, j in indices:
            if (i, j) in moved:
                continue
                
            state = self.grid[i, j]
            if state == 0 or self.mobility[state] == 0:
                continue

            # Calculate movement probability based on state and conditions
            move_prob = self.mobility[state] * social_distance_factor
            if state == self.INFECTED_NON_QUARANTINED:
                move_prob *= 0.5  # Reduced mobility for infected
                
            if random.random() < move_prob / 100:
                neighbors = self.get_neighbors(i, j)
                empty_neighbors = [(ni, nj) for ni, nj in neighbors if self.grid[ni, nj] == 0]
                
                if empty_neighbors:
                    ni, nj = random.choice(empty_neighbors)
                    new_grid[ni, nj] = state
                    new_grid[i, j] = 0
                    moved.add((ni, nj))
        
        self.grid = new_grid

    def try_infection(self, i: int, j: int, day: int):
        """Enhanced infection logic with realistic transmission dynamics."""
        if self.grid[i, j] != self.SUSCEPTIBLE:
            return
            
        # Calculate local density
        neighbors = self.get_neighbors(i, j)
        infected_neighbors = sum(1 for ni, nj in neighbors 
                               if self.grid[ni, nj] in [self.INFECTED_NON_QUARANTINED, 
                                                      self.INFECTED_QUARANTINED])
        
        # Density-dependent transmission
        density_factor = infected_neighbors / len(neighbors)
        seasonal_mod = self.get_seasonal_modifier(day)
        
        # Base transmission probability
        transmission_prob = 0
        
        for ni, nj in neighbors:
            if self.grid[ni, nj] == self.INFECTED_NON_QUARANTINED:
                transmission_prob += self.beta_n * seasonal_mod * (1 + density_factor)
            elif self.grid[ni, nj] == self.INFECTED_QUARANTINED:
                transmission_prob += self.beta_q * seasonal_mod * (1 + density_factor * 0.5)
        
        # Cap maximum transmission probability
        transmission_prob = min(0.95, transmission_prob)
        
        if random.random() < transmission_prob:
            # Dynamic quarantine probability based on current infection levels
            current_quarantine_prob = min(0.9, self.quarantine_ratio * (1 + density_factor))
            self.grid[i, j] = (self.INFECTED_QUARANTINED 
                              if random.random() < current_quarantine_prob 
                              else self.INFECTED_NON_QUARANTINED)

    def try_recovery(self, i: int, j: int, day: int):
        """Attempt to recover an infected individual with time-dependent rates."""
        # Improve recovery rates over time
        time_factor = min(1.5, 1 + (day / 100))  # Cap at 50% improvement
        
        if self.grid[i, j] == self.INFECTED_NON_QUARANTINED:
            if random.random() < (self.gamma_n * time_factor):
                self.grid[i, j] = self.RECOVERED_NON_QUARANTINED
        elif self.grid[i, j] == self.INFECTED_QUARANTINED:
            if random.random() < (self.gamma_q * time_factor):
                self.grid[i, j] = self.RECOVERED_QUARANTINED

    def step(self, day: int):
        """Perform one simulation step with time-dependent parameters."""
        self.random_walk(day)
        
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if self.grid[i, j] == self.SUSCEPTIBLE:
                    self.try_infection(i, j, day)
                elif self.grid[i, j] in [self.INFECTED_NON_QUARANTINED, 
                                       self.INFECTED_QUARANTINED]:
                    self.try_recovery(i, j, day)

    def get_statistics(self) -> dict:
        """Get current simulation statistics."""
        total_cells = self.grid_size * self.grid_size
        return {
            'Susceptible': np.sum(self.grid == self.SUSCEPTIBLE) / total_cells,
            'Active_NonQuarantined': np.sum(self.grid == self.INFECTED_NON_QUARANTINED) / total_cells,
            'Active_Quarantined': np.sum(self.grid == self.INFECTED_QUARANTINED) / total_cells,
            'Recovered_NonQuarantined': np.sum(self.grid == self.RECOVERED_NON_QUARANTINED) / total_cells,
            'Recovered_Quarantined': np.sum(self.grid == self.RECOVERED_QUARANTINED) / total_cells,
            'Empty': np.sum(self.grid == self.EMPTY) / total_cells
        }

    def visualize_grid(self, step_number: int = None):
        """Visualize current grid state."""
        colors = ['white', 'blue', 'red', 'purple', 'green', 'cyan']
        cmap = ListedColormap(colors)

        plt.figure(figsize=(8, 8))
        im = plt.imshow(self.grid, cmap=cmap)
        plt.gca().set_xticks([])
        plt.gca().set_yticks([])

        cbar = plt.colorbar(im, ticks=[0.4, 1.2, 2.0, 2.8, 3.6, 4.4])
        cbar.ax.set_yticklabels([
            'Empty',
            'Susceptible',
            'Non-quarantined Infected',
            'Quarantined Infected',
            'Recovered (Non-quarantined)',
            'Recovered (Quarantined)'
        ])

        title = 'Prediction Grid State'
        if step_number is not None:
            title += f' (Step {step_number})'
        plt.title(title)
        plt.tight_layout()

    def plot_prediction_comparison(self, sim_results: pd.DataFrame, save_plots: bool = True):
        """Plot prediction results against real data."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
        
        # Get normalization base
        total_population = self.country_data['Confirmed'].max()
        
        # Training period
        training_days = np.arange(len(self.training_data))
        
        # Testing period (start from last training point)
        n_pred_points = len(self.testing_data)
        test_days = np.arange(len(self.training_data) - 1, len(self.training_data) + n_pred_points - 1)
        
        # Debug print actual values being plotted
        print("\n[DEBUG] Plotting Data Verification:")
        print(f"Training end date: {self.training_data['Date'].iloc[-1]}")
        print(f"Testing start date: {self.testing_data['Date'].iloc[0]}")
        print(f"Last training point (Active): {self.training_data['Active'].iloc[-1]/total_population:.4f}")
        print(f"First testing point (Active): {self.testing_data['Active'].iloc[0]/total_population:.4f}")
        print(f"First prediction point (Active): {(sim_results['Active_NonQuarantined'].iloc[0] + sim_results['Active_Quarantined'].iloc[0]):.4f}")
        
        # Ensure prediction starts at the last training point value
        last_training_active = self.training_data['Active'].iloc[-1] / total_population
        last_training_recovered = self.training_data['Recovered'].iloc[-1] / total_population
        
        # Adjust first prediction point to match last training point
        sim_results.iloc[0, sim_results.columns.get_indexer(['Active_NonQuarantined', 'Active_Quarantined'])] = \
            [last_training_active * (1 - self.quarantine_ratio), last_training_active * self.quarantine_ratio]
        sim_results.iloc[0, sim_results.columns.get_indexer(['Recovered_NonQuarantined', 'Recovered_Quarantined'])] = \
            [last_training_recovered * (1 - self.quarantine_ratio), last_training_recovered * self.quarantine_ratio]
        
        # Plot Active Cases
        ax1.plot(training_days, 
                self.training_data['Active'] / total_population,
                'b-', label='Training Data (Active)', linewidth=2)
        ax1.plot(test_days,
                self.testing_data['Active'] / total_population,
                'g--', label='Real Data (Active)', linewidth=2)
        ax1.plot(test_days,
                sim_results['Active_NonQuarantined'].values[:n_pred_points] + 
                sim_results['Active_Quarantined'].values[:n_pred_points],
                'r-', label='Predicted Active', linewidth=2)
        
        # Plot Recovered Cases
        ax2.plot(training_days,
                self.training_data['Recovered'] / total_population,
                'b-', label='Training Data (Recovered)', linewidth=2)
        ax2.plot(test_days,
                self.testing_data['Recovered'] / total_population,
                'g--', label='Real Data (Recovered)', linewidth=2)
        ax2.plot(test_days,
                sim_results['Recovered_NonQuarantined'].values[:n_pred_points] + 
                sim_results['Recovered_Quarantined'].values[:n_pred_points],
                'r-', label='Predicted Recovered', linewidth=2)
        
        # Add vertical line to mark training/prediction split
        for ax in [ax1, ax2]:
            ax.axvline(x=len(self.training_data)-1, color='k', linestyle='--', alpha=0.5)
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
        # Get normalization base
        total_population = self.country_data['Confirmed'].max()
        
        # Ensure arrays are same length
        n_points = min(len(self.testing_data), len(sim_results))
        
        # Get real data
        real_active = self.testing_data['Active'].values[:n_points] / total_population
        real_recovered = self.testing_data['Recovered'].values[:n_points] / total_population
        
        # Get predicted data
        pred_active = (sim_results['Active_NonQuarantined'].values[:n_points] + 
                      sim_results['Active_Quarantined'].values[:n_points])
        pred_recovered = (sim_results['Recovered_NonQuarantined'].values[:n_points] + 
                         sim_results['Recovered_Quarantined'].values[:n_points])
        
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

    def get_seasonal_modifier(self, day: int):
        """Calculate seasonal modification to transmission rates."""
        # Assume annual cycle with peak in winter
        seasonal_factor = 1 + 0.2 * math.sin(2 * math.pi * day / 365)
        return seasonal_factor


def run_covid_prediction(data_path: str, country: str, start_date: str, end_date: str,
                        save_plots: bool = True):
    """
    Run COVID-19 prediction simulation.

    Parameters:
    -----------
    data_path: str
        Path to COVID-19 data CSV
    country: str
        Country to analyze
    start_date: str
        Start date in YYYY-MM-DD format
    end_date: str
        End date in YYYY-MM-DD format
    save_plots: bool
        Whether to save visualization plots
    """
    # Initialize simulator
    simulator = CovidPredictionSimulator(data_path, country, start_date, end_date)

    # Calculate prediction period (remaining days after training)
    n_prediction_days = (pd.to_datetime(end_date) - simulator.training_end_date).days

    # Store results
    results = []

    # Define visualization points
    viz_points = [
        n_prediction_days // 4,  # Quarter way
        n_prediction_days // 2,  # Halfway
        3 * n_prediction_days // 4,  # Three-quarters
        n_prediction_days - 1  # End
    ]

    print(f"\nStarting prediction simulation for {n_prediction_days} days...")

    # Run prediction
    for day in range(n_prediction_days):
        simulator.step(day)
        stats = simulator.get_statistics()
        results.append(stats)

        # Print progress
        if day % 10 == 0:
            print(f"Day {day}: Active cases = {(stats['Active_NonQuarantined'] + stats['Active_Quarantined']):.4f}")

        # Visualize at specific points
        if day in viz_points and save_plots:
            simulator.visualize_grid(step_number=day)
            plt.savefig(f'{country}_prediction_day_{day}.png',
                        dpi=300, bbox_inches='tight')
            plt.close()

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)

    # Plot comparison
    simulator.plot_prediction_comparison(results_df, save_plots = save_plots)

    # Calculate and print error metrics
    errors = simulator.calculate_prediction_error(results_df)
    print("\nPrediction Error Metrics:")
    for metric, value in errors.items():
        print(f"{metric}: {value:.4f}")

    return results_df, errors


if __name__ == "__main__":
    # Example usage
    results, errors = run_covid_prediction(
        data_path="covid_19_clean_complete.csv",
        country="Cambodia",
        start_date="2020-01-22",
        end_date="2020-05-22"  # First 60 days for training, rest for prediction
    )
