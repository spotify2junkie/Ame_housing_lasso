# Ame_housing

In this project I analyze housing data from ames county data.

# Part 1
- Running customerized LGBMRegressors iteratively(can choose iteration,such as 100,200,etc. Depends on the computer computing power) to pick the most important features, rank them in an decresing order.
- drop the zero-importance features selected by previous method
- use randomizedsearchCV and gridsearchCV to pick the optimized hyperparameters.

# Part 2 
- Writing one-step lasso `algorithm` to compute coefficents for regression result
