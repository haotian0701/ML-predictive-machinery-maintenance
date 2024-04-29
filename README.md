# Machine Learning for Predictive Failure in Industrial Machinery 

This project utilizes and contrasts several machine learning techniques -- including logistic regression, artificial neural networks (ANNs), and sequential ANNs -- to address the challenge of preemptive equipment failure detection in industrial machinery. By developing a comprehensive predictive failure system, this project showcases how to enhance safety and minimize economic losses in industrial settings, focusing on deriving actionable insights that promote proactive maintenance strategies.

To ensure the effectiveness of the models, this project addresses various challenges such as computational limitations, feature engineering, and potential overfitting. A structured train-validate-test framework is employed, focusing on metrics such as AUROC, accuracy, precision, and recall, _with some models achieving validation and test AUROC figures exceeding 0.999._ Through these efforts, the project demonstrates how advanced predictive techniques can minimize downtime, ensure operational safety, and extend the lifespan of industrial machinery.

To get started with running the code yourself, please see the [Usage section](#usage) below.

## Data and Preprocessing

...

## Methodology

...

## Results

...

## Discussion

...

## Usage

To properly utilize this project, ensure that Python can import from the project's `src/` and `data/` directories, necessary for the Jupyter notebooks to import the necessary modules and data files.

#### Environment Setup

First, set up the environment using Conda:

```bash
# create environment from project directory
$ conda env create -f environment.yml
$ conda activate ml-predictive-failure
```

#### Running the Notebooks

With the environment activated, launch Jupyter to run the notebooks:

```bash
(ml-predictive-failure) $ jupyter notebook
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
- Email: alexander.w.wilcox [at] gmail.com