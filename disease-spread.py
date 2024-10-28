# pip install numpy matplotlib

import numpy as np
import matplotlib.pyplot as plt

class DiseaseSpreadSimulator:
    def __init__(self, n_states: int = 5, population_size: int = 10):
        """
        Initialize the disease spread simulator
        """
        self.n_states = n_states
        self.population_size = population_size

        # Initialize the Markov model parameters
        self._initialize_model()

        print(f"Initialized DiseaseSpreadSimulator with {self.n_states} states, population size {self.population_size}")

    def _initialize_model(self):
        """Initialize the HMM with default parameters"""
        print("Initializing HMM model...")
        # Starting probabilities
        self.startprob = np.array([0.5, 0.25, 0.15, 0.1, 0.0])  # Adjusted for new state
        print(f"Start probabilities set to: {self.startprob}")

        # Transition matrix (S->E->I->R->D)
        self.transmat = np.array([
            [0.60, 0.20, 0.20, 0.0, 0.0],  # Susceptible -> [S, E, I, R, D]
            [0.0, 0.9, 0.1, 0.0, 0.0],    # Exposed -> [S, E, I, R, D]
            [0.0, 0.0, 0.8, 0.1, 0.1],  # Infected -> [S, E, I, R, D]
            [0.0, 0.0, 0.0, 1.0, 0.0],    # Recovered -> [S, E, I, R, D]
            [0.0, 0.0, 0.0, 0.0, 1.0],    # Dead -> [S, E, I, R, D]
        ])
        print(f"Transition matrix set to: \n{self.transmat}")

        # Emission probabilities (hidden state -> observed symptoms)
        self.emissionprob = np.array([
            [0.50, 0.25, 0.25],  # Susceptible -> [No symptoms, Mild, Severe]
            [0.1, 0.15, 0.75],   # Exposed -> [No symptoms, Mild, Severe]
            [0.1, 0.6, 0.3],     # Infected -> [No symptoms, Mild, Severe]
            [0.8, 0.15, 0.05],   # Recovered -> [No symptoms, Mild, Severe]
            [0.0, 0.0, 0.0],     # Dead -> [No symptoms, Mild, Severe]
        ])
        print(f"Emission probabilities set to: \n{self.emissionprob}")

        print("HMM model initialized.")

    def simulate(self, n_days: int) -> np.ndarray:
        """
        Simulate disease spread for given number of days and population
        """
        print(f"Starting simulation for {n_days} days...")

        # Initialize the arrays to hold the states for all individuals
        all_states = np.zeros((self.population_size, n_days), dtype=int)
        all_observations = np.zeros((self.population_size, n_days), dtype=int)

        for i in range(self.population_size):
            # Add randomness to start probabilities
            startprob = self.startprob + np.random.normal(0, 0.01, self.n_states)
            startprob = np.clip(startprob, 0, 1)
            startprob /= startprob.sum()
            startprob_cdf = np.cumsum(startprob)

            # Initialize the first state based on start probabilities
            r = np.random.rand()
            initial_state = np.searchsorted(startprob_cdf, r)
            states = [initial_state]

            # Add randomness to emission probabilities
            emissionprob = self.emissionprob + np.random.normal(0, 0.01, self.emissionprob.shape)
            emissionprob = np.clip(emissionprob, 0, 1)

            # Add a small epsilon to avoid division by zero
            epsilon = 1e-10
            row_sums = emissionprob.sum(axis=1, keepdims=True)
            emissionprob /= (row_sums + epsilon)

            emissionprob_cdf = np.cumsum(emissionprob, axis=1)

            # Generate the first observation
            r = np.random.rand()
            initial_observation = np.searchsorted(emissionprob_cdf[initial_state], r)
            observations = [initial_observation]

            for t in range(1, n_days):
                current_state = states[-1]

                # Add randomness to transition probabilities
                transmat = self.transmat + np.random.normal(0, 0.01, self.transmat.shape)
                transmat = np.clip(transmat, 0, 1)
                transmat /= transmat.sum(axis=1, keepdims=True)
                transmat_cdf = np.cumsum(transmat, axis=1)

                r = np.random.rand()
                next_state = np.searchsorted(transmat_cdf[current_state], r)
                states.append(next_state)

                # Generate observation based on the new state
                r = np.random.rand()
                next_observation = np.searchsorted(emissionprob_cdf[next_state], r)
                observations.append(next_observation)

            all_states[i, :] = states
            all_observations[i, :] = observations

            if (i + 1) % 100 == 0:
                print(f"Simulated {i+1} individuals")

        print("Simulation complete.")
        return all_states, all_observations

    def plot_simulation(self, states: np.ndarray, observations: np.ndarray):
        """
        Plot the simulation results
        """
        print("Plotting simulation results...")
        n_days = states.shape[1]
        state_labels = ['Susceptible', 'Exposed', 'Infected', 'Recovered', 'Dead']
        symptom_labels = ['No symptoms', 'Mild', 'Severe']

        # Initialize an array to hold counts per state per day
        state_counts = np.zeros((n_days, self.n_states))
        symptom_counts = np.zeros((n_days, len(symptom_labels)))

        # For each day, count the number of people in each state and symptom
        for day in range(n_days):
            for state in range(self.n_states):
                state_counts[day, state] = np.sum(states[:, day] == state)
            for symptom in range(len(symptom_labels)):
                symptom_counts[day, symptom] = np.sum(observations[:, day] == symptom)

        # Plot the state counts with different colors
        state_colors = ['blue', 'orange', 'green', 'red', 'black']
        plt.figure(figsize=(12, 6))
        for state in range(self.n_states):
            plt.plot(state_counts[:, state], label=state_labels[state], color=state_colors[state])

        plt.xlabel('Days')
        plt.ylabel('Number of People')
        plt.title('Disease Spread Simulation - States')
        plt.legend()
        plt.grid(True)
        plt.show()

        # Plot the symptom counts with different colors
        symptom_colors = ['gray', 'yellow', 'purple']
        plt.figure(figsize=(12, 6))
        for symptom in range(len(symptom_labels)):
            plt.plot(symptom_counts[:, symptom], label=symptom_labels[symptom], color=symptom_colors[symptom])

        plt.xlabel('Days')
        plt.ylabel('Number of People')
        plt.title('Disease Spread Simulation - Symptoms')
        plt.legend()
        plt.grid(True)
        plt.show()

        print("Plotting complete.")

    def forward_backward(self, observations: np.ndarray) -> np.ndarray:
        """
        Perform the forward-backward algorithm to compute smoothed probabilities
        of hidden states given the sequence of observations.
        """
        n_days = observations.shape[1]
        fwd = np.zeros((n_days, self.n_states))
        bkw = np.zeros((n_days, self.n_states))
        smoothed = np.zeros((n_days, self.n_states))

        # Forward pass
        fwd[0] = self.startprob * self.emissionprob[:, observations[0]]
        fwd[0] /= fwd[0].sum()  # Normalize

        for t in range(1, n_days):
            for j in range(self.n_states):
                fwd[t, j] = np.sum(fwd[t-1] * self.transmat[:, j]) * self.emissionprob[j, observations[t]]
            fwd[t] /= fwd[t].sum()  # Normalize

        # Backward pass
        bkw[-1] = 1  # Initialize with ones

        for t in range(n_days - 2, -1, -1):
            for i in range(self.n_states):
                bkw[t, i] = np.sum(self.transmat[i] * self.emissionprob[:, observations[t+1]] * bkw[t+1])
            bkw[t] /= bkw[t].sum()  # Normalize

        # Combine forward and backward probabilities
        for t in range(n_days):
            smoothed[t] = fwd[t] * bkw[t]
            smoothed[t] /= smoothed[t].sum()  # Normalize

        return smoothed

    def expectation_maximization(self, observations: np.ndarray, max_iter: int = 100, tol: float = 1e-4):
        """
        Perform the EM algorithm to estimate the parameters of the HMM.
        """
        n_days = observations.shape[1]
        n_individuals = observations.shape[0]

        # Initialize parameters
        transmat = self.transmat.copy()
        emissionprob = self.emissionprob.copy()

        for iteration in range(max_iter):
            # E-step: Calculate expected values
            fwd = np.zeros((n_individuals, n_days, self.n_states))
            bkw = np.zeros((n_individuals, n_days, self.n_states))
            gamma = np.zeros((n_individuals, n_days, self.n_states))
            xi = np.zeros((n_individuals, n_days - 1, self.n_states, self.n_states))

            # Forward-backward to calculate gamma and xi
            for i in range(n_individuals):
                fwd[i], bkw[i] = self._forward_backward(observations[i], transmat, emissionprob)
                gamma[i] = fwd[i] * bkw[i]
                gamma[i] /= gamma[i].sum(axis=1, keepdims=True)

                for t in range(n_days - 1):
                    xi[i, t] = fwd[i, t, :, None] * transmat * emissionprob[:, observations[i, t+1]] * bkw[i, t+1]
                    xi[i, t] /= xi[i, t].sum()

            # M-step: Update parameters
            transmat_new = xi.sum(axis=(0, 1)) / gamma[:, :-1].sum(axis=(0, 1))
            emissionprob_new = np.zeros_like(emissionprob)

            for k in range(self.n_states):
                for o in range(emissionprob.shape[1]):
                    mask = (observations == o)
                    emissionprob_new[k, o] = (gamma[:, :, k] * mask).sum() / gamma[:, :, k].sum()

            # Check for convergence
            if np.allclose(transmat, transmat_new, atol=tol) and np.allclose(emissionprob, emissionprob_new, atol=tol):
                break

            transmat, emissionprob = transmat_new, emissionprob_new

        self.transmat = transmat
        self.emissionprob = emissionprob

    def _forward_backward(self, observation, transmat, emissionprob):
        # Implement the forward-backward algorithm here
        # Return forward and backward probabilities
        pass

# Example usage
def run_simulation(n_days: int = 50, population_size: int = 2000):
    """
    Run a complete simulation with visualization
    """
    print("Initializing simulator...")
    simulator = DiseaseSpreadSimulator(population_size=population_size)

    print("Running simulation...")
    states, observations = simulator.simulate(n_days)

    print("Plotting results...")
    simulator.plot_simulation(states, observations)

    print("Simulation complete.")

    # Print final statistics
    final_day_states = states[:, -1]
    print("\nFinal Day Statistics:")
    print(f"Susceptible: {np.sum(final_day_states == 0)} people")
    print(f"Exposed: {np.sum(final_day_states == 1)} people")
    print(f"Infected: {np.sum(final_day_states == 2)} people")
    print(f"Recovered: {np.sum(final_day_states == 3)} people")
    print(f"Dead: {np.sum(final_day_states == 4)} people")

if __name__ == "__main__":
    run_simulation()
