# Multi-label feature selection via robust flexible sparse regularization (2023)

Authors = Liang Hu, Yonghao Li, and , Wanfu Gao,

Journal = Pattern Recognition.

Abstract :
Multi-label feature selection is an efficient technique to deal with the high dimensional multi-label data by selecting the optimal feature subset. Existing researches have demonstrated that l 1 -norm and l 2 , 1 - norm are 
promising roles for multi-label feature selection. However, two important issues are ignored when existing l 1 -norm and l 2 , 1 -norm based methods select discriminative features for multi-label data. First, l 1 -norm can 
enforce sparsity on each feature across all instances while numerous selected features lack discrimination due to the generated zero weight values. Second, l 2 , 1 -norm not only neglects label- specific features but also 
ignores the redundancy among features. To this end, we design a Robust Flexible Sparse Regularization norm (RFSR), furthermore, proposing a global optimization framework named Ro- bust Flexible Sparse regularized multi-label Feature Selection (RFSFS) based on RFSR. Finally, an efficient alternating multipliers based optimization scheme is developed to iteratively optimize RFSFS. Empirical studies on fifteen benchmark multi-label data sets 
demonstrate the effectiveness and efficiency of RFSFS.

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
MLFS-RFSFS.py
+ Run GPU
+ Code Developed Based on Updated Formulas from the Paper
+ Parameters Tuning
+ Calculate the Average Percentage for Each Selected Feature (1 to 20) over 5 Iterations
