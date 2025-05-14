## [Multi-label feature selection via robust flexible sparse regularization (2023)](https://www.sciencedirect.com/science/article/abs/pii/S0031320322005544)

Authors = Liang Hu, Yonghao Li, and , Wanfu Gao,

Journal = Pattern Recognition.

Abstract :
Multi-label feature selection is an efficient technique to deal with the high dimensional multi-label data by selecting the optimal feature subset. Existing researches have demonstrated that l 1 -norm and l 2 , 1 - norm are 
promising roles for multi-label feature selection. However, two important issues are ignored when existing l 1 -norm and l 2 , 1 -norm based methods select discriminative features for multi-label data. First, l 1 -norm can 
enforce sparsity on each feature across all instances while numerous selected features lack discrimination due to the generated zero weight values. Second, l 2 , 1 -norm not only neglects label- specific features but also 
ignores the redundancy among features. To this end, we design a Robust Flexible Sparse Regularization norm (RFSR), furthermore, proposing a global optimization framework named Ro- bust Flexible Sparse regularized multi-label Feature Selection (RFSFS) based on RFSR. Finally, an efficient alternating multipliers based optimization scheme is developed to iteratively optimize RFSFS. Empirical studies on fifteen benchmark multi-label data sets 
demonstrate the effectiveness and efficiency of RFSFS.

If you find this implementation helpful in your work, please consider citing both the original paper and our related research on multi-label feature selection:

Original Paper:
'''
@article{li2023multi,
  title={Multi-label feature selection via robust flexible sparse regularization},
  author={Li, Yonghao and Hu, Liang and Gao, Wanfu},
  journal={Pattern Recognition},
  volume={134},
  pages={109074},
  year={2023},
  publisher={Elsevier}
}
'''

##Our Paper:
'''
@article{faraji2024multi,
  title={Multi-label feature selection with global and local label correlation},
  author={Faraji, Mohammad and Seyedi, Seyed Amjad and Tab, Fardin Akhlaghian and Mahmoodi, Reza},
  journal={Expert Systems with Applications},
  volume={246},
  pages={123198},
  year={2024},
  publisher={Elsevier}
}
'''
