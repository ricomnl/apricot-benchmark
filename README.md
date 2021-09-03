# Apricot Benchmark

Goal: compare all the functions and optimizers across a few dozen data sets
Data sets: https://epistasislab.github.io/pmlb/#Dataset_format

Apply each objective and each optimizer to select a subset from each of the data sets. 
Compare optimizers using the objective function that they're attempting to optimize.
Can't really compare different objective functions that easily. 
    (1) for each objective/optimizer, look at how many examples are chosen from the smallest class, and 
    (2) train ML models using each subset and seeing how well they can make predictions on the test set.
    (3) compare the performance of the models with the ones trained on the full datasets 

Will the metadata from each data set have some association with which objectives or optimizers perform well, e.g., feature-based functions work well with categorical data but facility location works better when everything is float?

Challenge: data engineering is an important aspect of getting any ML method to work and it wouldn't be fair to the functions to simply use the data raw

Data Prep:
Filter by n_observations > 10_000: 417 -> 146 rows
Remove the feynman datasets: 146 -> 26 rows
13 Classification, 13 Regression datasets
- [ ] will have to look at the categorical datasets to see what kind of encoding we need