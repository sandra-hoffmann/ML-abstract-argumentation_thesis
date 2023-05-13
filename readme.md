# ML Abstract Argumentation

This project contains the code to my bachelor thesis: "Investigating the Influence of Graph Properties on the Prediction Quality of Machine Learning Methods in the Context of Abstract Argumentation".


### Prerequisites

Make sure you have the following prerequisites installed:

- Python (version 3.10)
- Required Python packages:
  - `numpy 1.24.3`
  - `scikit-learn 1.2.2`
  - `pandas 2.0.1`
  - `matplotlib 3.7.1`


## Usage

The project provides a command-line interface for running the classifiers. You can specify the path to the input data as well as the features to be tested and a time limit for the solver. The code will loop through the specified features

### Single Features 
```shell
python main.py -X path/to/training/data -y path/to/testing/data -features "[feature1, feature2, feature3]"
```

### Feature Combinations
```shell
python main.py -X path/to/training/data -y path/to/testing/data -timeout 600000 -features "[[feature1, feature2], [feature1, feature2, feature3]]"
```

- `-X`: Path to the training data
- `-y`: Path to the testing data
- `-timeout`: Timeout limit for solver (optional, default = 3600000)
- `-features`: Comma-separated list of features to use

The following features can be used:

- `degree`
- `katz_centrality`
- `page_rank`
- `closeness_centrality`
- `betweenness_centrality`
- `no_of_sccs`
- `scc_size`
- `strong_connectivity`
- `symmetry`
- `asymmetry`
- `irreflexive`
- `attacks_all_others`
- `avg_degree`
- `aperiodicity`

## Results

The results of the classifiers will be saved in a text file named `testruns/TEST_RESULTS.txt` in LaTex notation.


## Note

Due to the large amount of Ads in the PBBG Dataset it was not possible to upload the solutions and pkl files for this dataset.



