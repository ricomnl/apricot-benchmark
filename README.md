# Apricot Benchmark
## Setup
```
# Setup poetry
make .venv

# Run streamlit
make run
```

## Project
### Goal
Compare all the functions and optimizers across a few dozen data sets.
Apply each objective and each optimizer to select a subset from each of the data sets. 
Compare optimizers using the objective function that they're attempting to optimize.
Can't really compare different objective functions that easily. 
    (1) for each objective/optimizer, look at how many examples are chosen from the smallest class, and
    (2) train ML models using each subset and seeing how well they can make predictions on the test set.
    (3) compare the performance of the models with the ones trained on the full datasets 

Will the metadata from each data set have some association with which objectives or optimizers perform well, e.g., feature-based functions work well with categorical data but facility location works better when everything is float?

n_observations
n_features
number continuous/categorical/discrete (proportions?)
n_classes
type: binary, categorical, continuous
task: classification, regression
imbalance
average correlation coefficient

metrics:
number of observations chosen from minority class

What does "transform" and "code" mean?

- one hot encoding categoricals
- setting min value to 0

- [ ] set up richer metadata dataframe so we can compare 
- [ ] set up code for feature-based and facility location
- [ ] can we come up with a concave function that doesn't divide
- [ ] look at speed and memory cost using `sklearn.datasets.fetch_covtype` of increasing size
- [ ] look at julia priorityqueue
- [ ] implement pytorch version of feature-based function (naive)
- [ ] could show that we can train a model on imagenet using x% of the data

1. Quality of SO
2. Time and Memory
3. Quality of downstream ML (compared within and to full data ML)

### Data sets
https://epistasislab.github.io/pmlb/

### Functions
- Feature-based
- Maximum Coverage
- Facility Location
- Graph Cut
- Sum Redundancy
- Saturated Coverage
- Mixtures

### Optimizers
- Naive
- Lazy
- Two-stage
- Approximate Lazy
- Stochastic
- Sample
- GreeDi
- Modular Greedy
- Bidirectional

### Concave Function
- log
- sqrt
- sigmoid

### Ideas
- Compare on speed as well? (can just take 1 one of the bigger datasets)
- Test scalability of the different functions using the Feature-based optimizer

### Data Prep
Challenge: data engineering is an important aspect of getting any ML method to work and it wouldn't be fair to the functions to simply use the data raw

Filter by n_observations > 10_000: 417 -> 146 rows
Remove the feynman datasets: 146 -> 26 rows
13 Classification, 13 Regression datasets
- [ ] will have to look at the categorical datasets to see what kind of encoding we need
