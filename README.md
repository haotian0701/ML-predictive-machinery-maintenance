# Machine Learning for Predictive Failure in Industrial Machinery 

This project utilizes and contrasts several machine learning techniques — including logistic regression, artificial neural networks (ANNs), and sequential ANNs — to address the challenge of preemptive equipment failure detection in industrial machinery. Using the [_Microsoft Azure Predictive Maintenance_ dataset](https://www.kaggle.com/datasets/arnabbiswas1/microsoft-azure-predictive-maintenance/data), this project develops a comprehensive predictive failure system. It showcases how to enhance safety and minimize economic losses in industrial settings by deriving actionable insights that promote proactive maintenance strategies.

To ensure the effectiveness of the models, this project addresses various challenges such as computational limitations, feature engineering, and potential overfitting. A structured train-validate-test framework is employed, focusing on metrics such as AUROC, accuracy, precision, and recall, _with some models achieving validation and test AUROC figures exceeding 0.999._ Through these efforts, the project demonstrates how advanced predictive techniques can minimize downtime, ensure operational safety, and extend the lifespan of industrial machinery.

To get started with running the code yourself, please see the [Usage section](#usage) below.

## Data and Preprocessing

This project uses the [_Microsoft Azure Predictive Maintenance_ dataset](https://www.kaggle.com/datasets/arnabbiswas1/microsoft-azure-predictive-maintenance/data) which includes machine data over the year 2015 from 5 different sources: maintenance records, failure history, error logs, static machine attributes such as model and age, and hourly telemetry data (hourly readings on voltage, rotation, pressure, and vibration). These data sources are preprocessed to merge into a unified dataset where each machine is represented by hourly records over the year 2015.

- <ins>Feature Data:</ins> The data is structured into discrete 24-hour windows to effectively model both static and dynamic aspects of machine performance. Experimentation with different time windows (12-, 48-, 72-, and 168-hours) confirmed that a 24-hour window was the smallest interval that achieved consistent, high-quality model performance, beyond which longer windows offered equivalent (or inferior) performance.
  - **Feedforward Models:** For logistic regression and MLPs, features within each window are summarized into a single vector using statistical measures such as mean, standard deviation, skewness, kurtosis, etc., resulting in 45 features per window.
  - **Sequential Models:** In contrast, RNN, LSTM, and GRU process each 24-hour window as a sequence of hourly snapshots, with 25 features per hour to maintain temporal relationships. 
- <ins>Target Data:</ins> Output labels are generated to indicate whether a machine will fail within the subsequent 6- or 24-hours, enabling the models to predict short-term and medium-term failures. 

To ensure accurate training and evaluation, features and labels are carefully aligned to avoid data leakage.

## Methodology

To ensure thorough evaluation and prevent overfitting, the dataset was partitioned randomly with 60% of data for training, 20% for validation, and 20% for testing. Each "example" corresponds to a 24-hour period for a given machine, and whether it failed in the subsequent 24-hour period. For ANNs, training was conducted over 200 epochs using binary cross entropy to optimize the models effectively without overfitting.

- _Logistic Regression_ served primarily for illustrative purposes to establish a baseline, demonstrating the non-linear relationship in the dataset where simpler linear models might fail.
- _MLP and Sequential Models (RNN, LSTM, GRU)_ were fine-tuned to optimize parameters like learning rate, batch size, and dropout rates over several experiments (not all included in notebooks). For sequential models, specific attention was given to RNN dropout and hidden layers, crucial for handling temporal data efficiently.

## Results

...

## Discussion

...

## Model Performance Snapshots

...

## Usage

To properly utilize this project, ensure that Python can import from the project's `src/` and `data/` directories, necessary for the Jupyter notebooks to import the necessary modules and data files.

##### Environment Setup

First, set up the environment using Conda:

```bash
# create environment from project directory
$ conda env create -f environment.yml
$ conda activate ml-pred-fail
```

##### Running the Notebooks

With the environment activated, launch Jupyter to run the notebooks:

```bash
(ml-pred-fail) $ jupyter notebook
```

This will start Jupyter. Open and execute the project notebooks to explore the code. When Jupyter is launched from the root of the project directory, it automatically configures the path to include the `src/` and `data/` directories, enabling straightforward imports within the notebooks.

## Areas for Future Improvement

- [ ] Refine models to predict _individual_ failure codes (rather than just any failure) to enhance model robustness.
- [ ] Investigate hybrid models that combine MLPs for "static" features (e.g., machine make) and RNNs for temporal data (e.g., rotation) to improve prediction accuracy.
- [ ] Explore advanced ensemble methods to improve performance.

## Acknowledgements 

- [_Microsoft Azure Predictive Maintenance_ dataset](https://www.kaggle.com/datasets/arnabbiswas1/microsoft-azure-predictive-maintenance/data).
- Riccardo Prosdocimi, my partner for this project.
- Professor Paul Hand, my Machine Learning professor.

## Contact Information

- Alexander Wilcox
- Email: alexander.w.wilcox [at] gmail [dot] com