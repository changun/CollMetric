# Last FM

## Auth
* Application name	RecSys
* API key	84e85abb3f7f44f9b67bc7f3031e3be6
* Shared secret	84b4180d8e60de7a6f71fab57567125d
* Registered to	changun423

## Results


Mathod | Recall@100 | Recall@20 | Precision@10 | Precision@1 | AUC
--- |   --- | --- | ---
Gaussian 50 K=3 |  0.211852229425 | 0.0774401503011 |0.255249745158| 0.3058103975535168 |0.916184575257
Gaussian 50 (l_b=0.01, l_cov=100, max_norm=1.7)|  0.2039 | 0.070009 |0.2212| 0.2477 | 0.910698
Gaussian 50 K=2 (l_b=0.01, l_cov=100, max_norm=1.7, mu=0.5)|  0.21084 | 0.07723 |0.25239| 0.31294597 | 0.91468
Gaussian 150 (Norm=1.7, Cov=10, l_var=1 ) | 0.190619453975 |0.0709777403096| 0.241366972477| 0.323139653415 | 0.910476768686
Gaussian 150 K=2 (Norm=1.7, Cov=10, l_var=1, l_den=0.1, mu=0.5 ) | 0.21794 |  0.07816074188772 | 0.26034658511| 0.3445 | 0.9153
Gaussian 150 K=3 (Norm=1.7, Cov=10, l_var=1, l_den=1, mu=0.7 ) | 0.22144 |  0.08246 | 0.27064 | 0.3425 | 0.913924
WARP=150 | 0.208313801473 | 0.0738431001838 | 0.240978593272 | 0.313965341488 | 0.912157401925
WARP=50 | 0.18878941080516173 |  0.065487667091223029 | 0.22660550458715598 | 0.30275229357798167 |  0.91106414275846248
 (l_u=0.01, l_v=0.01, l_b=0.0,)| 0.1860|0.0659|0.230|0.30173|

1. Covariance 
    * Recall@100 0.197319792341
    * Precision@1 0.339449541284
    * Precision@10 0.240672782875

2. K=3
    * Run K=1 with l_var = 1, l_v = 0.01, l_b=0.1, lu, then run K=3 with l_var = l_den = l_cov = 0.1

flickr_50 = KBPRModel(50, n_users, n_items, batch_size=100000,     per_user_sample=50, learning_rate=0.1, lambda_u=0.0, lambda_v=0.0, lambda_bias=1, use_bias=True, K=1, margin=1, variance_mu=1, update_mu=True, normalization=False, uneven_sample=True, lambda_variance=1, lambda_cov=1, update_density=False, use_warp=True, max_norm=1.2)


*LightFM*
In_Matrix
[[0.4184033299493955, 0.0034447190935106798],
 [0.3044143264491691, 0.0031064822986305945],
 [0.11644544001749885, 0.001894023078518197]]
 Cold
 [[0.30248345245830349, 0.0035094230717956426],
 [0.20908348260091567, 0.0029798247360802201],
 [0.066909801707107197, 0.0015345875369843322]]
 *UV*
 In_Matrix
 [[0.41824902694675653, 0.0034424299433526268],
 [0.30390387230636201, 0.0031201847314594636],
 [0.1234482061983997, 0.0019566477395066171]]
 Cold
 [[0.29365484401860936, 0.0034441493618971942],
 [0.20075120330644347, 0.0029128143202748306],
 [0.067410042229556516, 0.0015154210915732525]]

 
 




# Flickr
## 100000 (reduced)

Mathod | Recall@100 | Recall@20 | Recall@10 | Precision@1 | AUC
--- |   --- | --- | ---
Gaussin 150 (Norm=1.2, Cov=10)|  0.0780040303029| | |  |0.865300
WARP=150 | 0.0746346305254
WARP=50 |  0.0598015873016
Gaussian 50| 0.0617738095238,
Gaussian 50 (norm=1.1, margin=0.5) | 0.065232884| 0.0235424| 0.0149113
Gaussian 50 (v_off=0.1) | 0.065567044| 0.0247426| 0.0156241
Gaussian 50 K=3 (mu=0.7, l_var=0.1, margin=1.0) | 0.07555024236
Gaussian 50 K=3 (mu=0.7, l_var=0.1, v_off=0.2, Margin=1.0)| 0.079795559105| 0.0301938




Mathod | Recall@100 | Recall@50 | Recall@10 | Cold@100 | Cold@50 | Cold@10
--- |   --- | --- | ---
Gaussian K=1, margin=0.5, v_off=0.3, weight_l1=0.0000001| 0.082058 | 0.05500 | 0.0231142 | 0.0397246 | 0.02113 | 0.0047
Gaussian K=3, margin=0.7, mu=1 | 0.086936 | 0.06028 | 0.025784 | 0.045313 | 0.02414 | 0.00510
WARP | 0.06153 |  0.0427710 | 0.0179320

# Movielens
# Rating >= 4
Mathod | Recall@100 | Recall@20 | Precision@10 | Precision@1 | AUC
--- |   --- | --- | ---
Gaussin 150 (Norm=1.5, Cov=10, l_var=0.1)|  | |0.11939495798319329 |  |0.94584455213735696
WARP=150 | ||0.125058823529||0.945631823207


Mathod | Recall@100 | Cold Recall@100 | Cold Recall@10| Precision@10 | Precision@1 | AUC
--- |   --- | --- | ---
Gaussin 50 (l_v_f=0.5, max_norm=1.1, norm_f=False, margin=0.5)| 0.617403413 | 0.20373126 | 0.0366411 
Gaussin 50 (K=2, margin=1.0, l_var=0.1, mu=0.7)| 0.6209177  |0.2223 |0.039507 | 
WARP l_v=1E-5| 0.621379
WARP w/ features| | 


In [17]: m.recall(cold_dict_m, None, exclude_items=list(in_m))
0%                          100%
[##############################] | ETA: 00:00:00 | Item ID: 0.313467193829
Total time elapsed: 00:02:47
Out[17]:
[[0.31346719382883104, 0.0011715261231483899],
 [0.21069601086004744, 0.00099863311826728167],
 [0.072576279067507923, 0.00058880582710675664]]

In [18]: m.recall(test_dict_b, exclude_dict_b, exclude_items=list(cold_b))
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
 in ()
----> 1 m.recall(test_dict_b, exclude_dict_b, exclude_items=list(cold_b))

NameError: name 'test_dict_b' is not defined

In [19]: m.recall(test_dict_m, exclude_dict_m, exclude_items=list(cold_m))
0%                          100%
[##############################] | ETA: 00:00:00 | Item ID: 0.617515539559
Total time elapsed: 00:02:40
Out[19]:
[[0.61751553954963645, 0.0011102291449725343],
 [0.47666021728983582, 0.0011604891678566177],
 [0.19236575586748075, 0.00087572433653511567]]


# BookCX
## > 20 mentions, > 5 ratings

Mathod | Recall@100 | Recall@50|Recall@20 | Precision@10 | Precision@1 | AUC
--- |   --- | --- | ---
WARP=50 | 0.2344| 0.15979  | 0.08979  | 0.0174 | 0.0310 | 0.841335 

Best Validation: -0.249742

## > 5 mentions, > 5 ratings (Validation)

Mathod | Recall@100 | Recall@50|Recall@20 | Precision@10 | Precision@1 | AUC
--- |   --- | --- | ---
WARP=50 | 0.14531|   |   | | | 
Gaussian 50 l_b 0.5 per_u 50 l_cov 10 l_den 0.1 norm 1.3| 0.141334 |  0.095013 | 0.05196564
Gaussian 50 K=2 mu=0.7| 0.15285 | 0.105840 |  0.061542 | 0.01332 | 0.0190
Gaussian 50 S=1, v_off=1, avg_f | 0.0818904 | 0.05171616| 0.0263583
0.1480808

Method | Recall@100 | Recall@50 | Recall@10 | Cold@100 | Cold@50 | Cold@10
--- |   --- | --- | ---
Gaussian K=1, v_off=0.3 | 0.1365612 | 0.0896194 | 0.0307918 | 0.1055654 | 0.0630525 | 0.0168371
Gaussian K=3, margin=1 | 0.14542 | 0.0958351 | 0.0349891 | 0.113734 | 0.0679328 |0.0183252
WARP | 0.1195536 | 0.077485 | 0.0263996In [82]:


m.recall(test_dict_b, exclude_dict_b, exclude_items=list(cold_b), n_users=3000)
0%                          100%
[##############################] | ETA: 00:00:00 | Item ID: 0.1297157953654
Total time elapsed: 00:00:26
Out[82]:
[[0.12971579536458247, 0.0049052995084884138],
 [0.086397824668718884, 0.0041357198726723315],
 [0.030369413295312293, 0.0025738049680791683]]

Out[142]:
[[0.096691519062868908, 0.0039953477055576312],
 [0.065958648320334906, 0.0033305888738790739],
 [0.023577678200482049, 0.0020604166733888315]]




