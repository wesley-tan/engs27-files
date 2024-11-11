# COVID-19 Prediction Simulation using Random Walk and SEIRD Model

This project simulates the spread of COVID-19 using a combination of a grid-based random walk and SEIRD model, implemented in Python. The simulation is based on data for a specified country and time period, allowing the user to visualize the spread dynamics and compare predicted values to actual historical data.

## Getting Started

### Prerequisites

To run this simulation, you need:

- Python 3.x
- Jupyter Notebook
- Required libraries:
  - `numpy`
  - `pandas`
  - `matplotlib`
  - `scipy`

These libraries can be installed using the following command:

```bash
pip install numpy pandas matplotlib scipy
```

### Dataset

The simulation requires a CSV file containing COVID-19 data with columns for `Date`, `Country/Region`, `Confirmed`, `Deaths`, and `Recovered`. The dataset used in the code is typically sourced from Kaggle or other COVID-19 open datasets.

### Running the Simulation

1. **Open the Notebook**: Open the provided notebook file, `engs-27-random-walk-and-seird-for-covid.ipynb`, in Jupyter Notebook.

2. **Configure Parameters**: In the notebook, configure the parameters such as:
   - `data_path`: Path to the COVID-19 data CSV file (e.g., `"/kaggle/input/covid-19-clean-complete/covid_19_clean_complete.csv"`).
   - `country`: The country for which you want to run the simulation (e.g., `"Cambodia"`).
   - `start_date` and `end_date`: The start and end dates for the simulation period (e.g., `"2020-01-22"` to `"2020-05-22"`).

3. **Run the Simulation**: Execute the cells in the notebook sequentially. The code will:
   - Load and process the COVID-19 data.
   - Initialize and run a COVID-19 prediction simulation using a grid-based random walk.
   - Calculate and visualize the grid states at specific days (e.g., days 15, 30, 45, and 60).
   - Generate plots comparing the predicted active and recovered cases to real data.
   - Display error metrics for model evaluation.

4. **Visualize Results**: The simulation will save and display the grid plots for days 15, 30, 45, and 60, along with a comparison of active and recovered cases.

### Sample Code Execution

To run the simulation in a standalone Python script (optional), use the following code template:

```python
if __name__ == "__main__":
    results, errors = run_covid_prediction(
        data_path="path_to_your_csv_file.csv",
        country="Country_Name",
        start_date="YYYY-MM-DD",
        end_date="YYYY-MM-DD"
    )
```

Replace `"path_to_your_csv_file.csv"`, `"Country_Name"`, `"YYYY-MM-DD"` with appropriate values.

## Outputs

The simulation generates the following outputs:

- **Grid State Visualizations**: Grid plots for days 15, 30, 45, and 60, saved as images (e.g., `Country_Name_prediction_day_15.png`).
- **Comparison Plots**: Plots showing the predicted vs. actual active and recovered cases over the simulation period.
- **Error Metrics**: Mean Squared Error (MSE), Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), and R² for both active and recovered cases.
