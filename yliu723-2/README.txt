I. Tools:
1. The analysis was done with the ABAGAIL package, which can be downloaded from https://github.com/pushkar/ABAGAIL.
2. The plots were generated using python matplotlib. 

II. Dataset and Randomized Optimization problems:
1.  The dataset used for finding weights for a neural network is the Breast-Cancer-Wisconsin data as used in Assignment1.  The data has been preprocessed to remove entries with missing values.  The data file is included with this submission with the name "breast-cancer-wisconsin.data"
2.  The three optimization problems used for testing the four search algorithms (RHC, SA, GA and MIMIC) are: (i) One-Max (ii) Traveling salesman (iii) Knapsack.

III. Code files:
1. matplot.py (plotting for figures in the analysis file with maplotlib)
2. OneMax_iteration (OneMax problem analyzed with RHC, SA, GA and MIMIC with different iteration #)
3. OneMax_optimized_timing.py (OneMax problem analyzed with RHC, SA, GA and MIMIC with optimized iteration # and timing for tables in the analysis)
4. knapsack_iteration.py (knapsack problem analyzed with RHC, SA, GA and MIMIC with different iteration #)
5. knapsack_optimized_timing.py (knapsack problem analyzed with RHC, SA, GA and MIMIC with optimized iteration # and timing for tables in the analysis)
6. travelingsalesman_iteration.py (travelingsalesman problem analyzed with RHC, SA, GA and MIMIC with different iteration #)
7. travelingsalesman_optimized_timing.py (travelingsalesman problem analyzed with RHC, SA, GA and MIMIC with optimized iteration # and timing for tables in the analysis)
8. CancerSet_NNweights_iteration (Wisconsin-breast-cancer dataset analyzed neural network weights optimized by RHC, SA and GA and MIMIC with different iteration #)
9. CancerSet_NNweights_trainingSize (Wisconsin-breast-cancer dataset analyzed neural network weights optimized by RHC, SA and GA and MIMIC with different training sample #)
