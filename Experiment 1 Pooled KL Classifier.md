panke66@scc-513 ldm_project]$ python "Joint difference test.py"
[Device] cuda
Traceback (most recent call last):
  File "/projectnb/cs790/students/panke66/ldm_project/Joint difference test.py", line 958, in <module>
    main()
  File "/projectnb/cs790/students/panke66/ldm_project/Joint difference test.py", line 883, in main
    meta, skipped = build_metadata(CONFIG)
  File "/projectnb/cs790/students/panke66/ldm_project/Joint difference test.py", line 154, in build_metadata
    df = pd.read_excel(excel_path)
  File "/share/pkg.8/python3/3.10.12/install/lib/python3.10/site-packages/pandas/io/excel/_base.py", line 478, in read_excel
    io = ExcelFile(io, storage_options=storage_options, engine=engine)
  File "/share/pkg.8/python3/3.10.12/install/lib/python3.10/site-packages/pandas/io/excel/_base.py", line 1500, in __init__
    raise ValueError(
ValueError: Excel file format cannot be determined, you must specify an engine manually.
[panke66@scc-513 ldm_project]$ file hand.xlsx
hand.xlsx: ASCII text, with very long lines, with CRLF line terminators
[panke66@scc-513 ldm_project]$ sed -i 's/pd.read_excel(excel_path)/pd.read_csv(excel_path)/' "Joint difference test.py"
[panke66@scc-513 ldm_project]$ python "Joint difference test.py"
[Device] cuda
[Metadata] usable images: 13772
[Metadata] skipped images: 27288
[Metadata] first few skipped files: [('9155218_dip2.png', 'joint_not_in_target_pip2_5'), ('9627893_dip2.png', 'joint_not_in_target_pip2_5'), ('9438149_dip3.png', 'joint_not_in_target_pip2_5'), ('9467722_dip2.png', 'joint_not_in_target_pip2_5'), ('9995277_dip2.png', 'joint_not_in_target_pip2_5'), ('9062483_dip3.png', 'joint_not_in_target_pip2_5'), ('9927318_mcp2.png', 'joint_not_in_target_pip2_5'), ('9540883_dip5.png', 'joint_not_in_target_pip2_5'), ('9456244_mcp5.png', 'joint_not_in_target_pip2_5'), ('9796299_dip5.png', 'joint_not_in_target_pip2_5')]

[Metadata] KL distribution:
kl
0    9776
1    2217
2    1665
3      69
4      45
Name: count, dtype: int64

[Metadata] Joint distribution:
joint
pip2    3455
pip4    3447
pip5    3440
pip3    3430
Name: count, dtype: int64

[Metadata] Joint x KL table:
kl        0    1    2   3   4
joint
pip2   2711  352  373  12   7
pip3   2344  575  474  23  14
pip4   2164  839  410  19  15
pip5   2557  451  408  15   9

[Split Summary]
Train patients: 2468 images: 9632
Val patients  : 530 images: 2068
Test patients : 530 images: 2072
Patient overlap: 0

======================================================================
Experiment 1: Pooled KL Classifier
======================================================================
Epoch 01 | train_loss=0.7159 train_acc=0.6768 train_f1=0.6770 | val_loss=0.7632 val_acc=0.6678 val_f1=0.4919
Epoch 02 | train_loss=0.5079 train_acc=0.7746 train_f1=0.7724 | val_loss=0.7293 val_acc=0.6939 val_f1=0.5498
Epoch 03 | train_loss=0.4401 train_acc=0.8051 train_f1=0.8015 | val_loss=0.8179 val_acc=0.6422 val_f1=0.5133
Epoch 04 | train_loss=0.4188 train_acc=0.8141 train_f1=0.8154 | val_loss=0.5924 val_acc=0.7548 val_f1=0.5153
Epoch 05 | train_loss=0.4001 train_acc=0.8269 train_f1=0.8285 | val_loss=0.6556 val_acc=0.7244 val_f1=0.5656
Epoch 06 | train_loss=0.3666 train_acc=0.8400 train_f1=0.8399 | val_loss=0.6276 val_acc=0.7558 val_f1=0.5570
Epoch 07 | train_loss=0.3442 train_acc=0.8533 train_f1=0.8551 | val_loss=0.6039 val_acc=0.7669 val_f1=0.5303
Epoch 08 | train_loss=0.3255 train_acc=0.8652 train_f1=0.8652 | val_loss=0.6595 val_acc=0.7456 val_f1=0.5267
Epoch 09 | train_loss=0.3087 train_acc=0.8696 train_f1=0.8708 | val_loss=0.7101 val_acc=0.7195 val_f1=0.5230
Epoch 10 | train_loss=0.2956 train_acc=0.8748 train_f1=0.8748 | val_loss=0.9151 val_acc=0.6330 val_f1=0.4487
Epoch 11 | train_loss=0.2742 train_acc=0.8870 train_f1=0.8875 | val_loss=0.6779 val_acc=0.7597 val_f1=0.5493
Epoch 12 | train_loss=0.2678 train_acc=0.8906 train_f1=0.8928 | val_loss=0.7155 val_acc=0.7519 val_f1=0.5211
[Best model saved] exp_outputs/pooled_kl_classifier.pt | best_val_f1=0.5656

[Test Result]
test_loss=0.6546 test_acc=0.7268 test_f1=0.4666

Classification Report:
              precision    recall  f1-score   support

           0     0.8923    0.8044    0.8461      1493
           1     0.3291    0.4875    0.3929       320
           2     0.5951    0.5927    0.5939       248
           3     0.0000    0.0000    0.0000         7
           4     0.5000    0.5000    0.5000         4
    
    accuracy                         0.7268      2072
   macro avg     0.4633    0.4769    0.4666      2072
weighted avg     0.7660    0.7268    0.7424      2072

Confusion Matrix:
[[1201  240   52    0    0]
 [ 123  156   41    0    0]
 [  22   78  147    0    1]
 [   0    0    6    0    1]
 [   0    0    1    1    2]]

======================================================================
Experiment 2: Pooled KL Classifier + Joint Index
======================================================================
Epoch 01 | train_loss=0.7698 train_acc=0.6589 train_f1=0.6535 | val_loss=0.7536 val_acc=0.6915 val_f1=0.5148
Epoch 02 | train_loss=0.4917 train_acc=0.7827 train_f1=0.7786 | val_loss=0.6907 val_acc=0.7036 val_f1=0.4987
Epoch 03 | train_loss=0.4489 train_acc=0.8041 train_f1=0.8030 | val_loss=0.6039 val_acc=0.7447 val_f1=0.5792
Epoch 04 | train_loss=0.4173 train_acc=0.8145 train_f1=0.8124 | val_loss=0.7125 val_acc=0.6741 val_f1=0.5408
Epoch 05 | train_loss=0.4150 train_acc=0.8216 train_f1=0.8214 | val_loss=0.6339 val_acc=0.7292 val_f1=0.5106
Epoch 06 | train_loss=0.3888 train_acc=0.8340 train_f1=0.8353 | val_loss=0.6732 val_acc=0.7224 val_f1=0.5275
Epoch 07 | train_loss=0.3683 train_acc=0.8417 train_f1=0.8412 | val_loss=0.6426 val_acc=0.7500 val_f1=0.5290
Epoch 08 | train_loss=0.3504 train_acc=0.8541 train_f1=0.8549 | val_loss=0.7115 val_acc=0.6978 val_f1=0.5497
Epoch 09 | train_loss=0.3152 train_acc=0.8683 train_f1=0.8671 | val_loss=0.6876 val_acc=0.7142 val_f1=0.5361
Epoch 10 | train_loss=0.3012 train_acc=0.8806 train_f1=0.8789 | val_loss=0.8130 val_acc=0.6596 val_f1=0.5292
Epoch 11 | train_loss=0.2917 train_acc=0.8809 train_f1=0.8805 | val_loss=0.6755 val_acc=0.7495 val_f1=0.4978
Epoch 12 | train_loss=0.2909 train_acc=0.8838 train_f1=0.8825 | val_loss=0.7339 val_acc=0.7147 val_f1=0.5356
[Best model saved] exp_outputs/pooled_plus_joint_kl_classifier.pt | best_val_f1=0.5792

[Test Result]
test_loss=0.6188 test_acc=0.7466 test_f1=0.5864

Classification Report:
              precision    recall  f1-score   support

           0     0.8886    0.8332    0.8600      1493
           1     0.3499    0.3750    0.3620       320
           2     0.5563    0.7177    0.6268       248
           3     0.4000    0.2857    0.3333         7
           4     0.7500    0.7500    0.7500         4
    
    accuracy                         0.7466      2072
   macro avg     0.5889    0.5923    0.5864      2072
weighted avg     0.7637    0.7466    0.7532      2072

Confusion Matrix:
[[1244  178   71    0    0]
 [ 133  120   67    0    0]
 [  23   45  178    2    0]
 [   0    0    4    2    1]
 [   0    0    0    1    3]]

======================================================================
Experiment 3: Joint Classifier (predict pip2/pip3/pip4/pip5)
======================================================================
Epoch 01 | train_loss=0.4513 train_acc=0.8213 train_f1=0.8209 | val_loss=0.2727 val_acc=0.8980 val_f1=0.8989
Epoch 02 | train_loss=0.2181 train_acc=0.9174 train_f1=0.9173 | val_loss=0.1352 val_acc=0.9536 val_f1=0.9536
Epoch 03 | train_loss=0.1481 train_acc=0.9477 train_f1=0.9476 | val_loss=0.1096 val_acc=0.9603 val_f1=0.9603
Epoch 04 | train_loss=0.1315 train_acc=0.9514 train_f1=0.9514 | val_loss=0.1074 val_acc=0.9599 val_f1=0.9596
Epoch 05 | train_loss=0.0880 train_acc=0.9672 train_f1=0.9672 | val_loss=0.0943 val_acc=0.9715 val_f1=0.9716
Epoch 06 | train_loss=0.0778 train_acc=0.9722 train_f1=0.9722 | val_loss=0.0573 val_acc=0.9816 val_f1=0.9816
Epoch 07 | train_loss=0.0759 train_acc=0.9709 train_f1=0.9709 | val_loss=0.0727 val_acc=0.9681 val_f1=0.9679
Epoch 08 | train_loss=0.0703 train_acc=0.9730 train_f1=0.9730 | val_loss=0.0499 val_acc=0.9797 val_f1=0.9797
Epoch 09 | train_loss=0.0490 train_acc=0.9834 train_f1=0.9834 | val_loss=0.0758 val_acc=0.9705 val_f1=0.9705
Epoch 10 | train_loss=0.0543 train_acc=0.9819 train_f1=0.9819 | val_loss=0.1001 val_acc=0.9623 val_f1=0.9623
Epoch 11 | train_loss=0.0510 train_acc=0.9827 train_f1=0.9827 | val_loss=0.0734 val_acc=0.9768 val_f1=0.9768
Epoch 12 | train_loss=0.0459 train_acc=0.9828 train_f1=0.9828 | val_loss=0.1780 val_acc=0.9396 val_f1=0.9392
[Best model saved] exp_outputs/joint_classifier.pt | best_val_f1=0.9816

[Test Result]
test_loss=0.0778 test_acc=0.9764 test_f1=0.9763

Classification Report:
              precision    recall  f1-score   support

           0     0.9942    0.9828    0.9884       522
           1     0.9672    0.9672    0.9672       519
           2     0.9615    0.9689    0.9652       515
           3     0.9826    0.9864    0.9845       516
    
    accuracy                         0.9764      2072
   macro avg     0.9764    0.9763    0.9763      2072
weighted avg     0.9764    0.9764    0.9764      2072

Confusion Matrix:
[[513   9   0   0]
 [  3 502  14   0]
 [  0   7 499   9]
 [  0   1   6 509]]

======================================================================
Experiment 4: Per-joint KL Classifiers
======================================================================

--- Per-joint KL experiment: pip2 ---
Epoch 01 | train_loss=0.7765 train_acc=0.6573 train_f1=0.6564 | val_loss=0.7648 val_acc=0.7154 val_f1=0.3211
Epoch 02 | train_loss=0.4994 train_acc=0.7808 train_f1=0.7854 | val_loss=0.7306 val_acc=0.7000 val_f1=0.3678
Epoch 03 | train_loss=0.4228 train_acc=0.8168 train_f1=0.8164 | val_loss=0.9099 val_acc=0.6135 val_f1=0.3329
Epoch 04 | train_loss=0.3640 train_acc=0.8421 train_f1=0.8360 | val_loss=0.9435 val_acc=0.5788 val_f1=0.3432
Epoch 05 | train_loss=0.3436 train_acc=0.8620 train_f1=0.8604 | val_loss=0.7055 val_acc=0.7365 val_f1=0.3189
Epoch 06 | train_loss=0.2937 train_acc=0.8844 train_f1=0.8848 | val_loss=0.6271 val_acc=0.7615 val_f1=0.3971
Epoch 07 | train_loss=0.2767 train_acc=0.8985 train_f1=0.8979 | val_loss=0.7649 val_acc=0.6981 val_f1=0.2938
Epoch 08 | train_loss=0.2613 train_acc=0.8968 train_f1=0.8979 | val_loss=0.6689 val_acc=0.7769 val_f1=0.3216
Epoch 09 | train_loss=0.2167 train_acc=0.9117 train_f1=0.9136 | val_loss=1.1277 val_acc=0.6154 val_f1=0.3006
Epoch 10 | train_loss=0.1948 train_acc=0.9237 train_f1=0.9229 | val_loss=0.7721 val_acc=0.7827 val_f1=0.3023
Epoch 11 | train_loss=0.1772 train_acc=0.9333 train_f1=0.9342 | val_loss=0.7398 val_acc=0.7712 val_f1=0.3359
Epoch 12 | train_loss=0.1540 train_acc=0.9416 train_f1=0.9427 | val_loss=0.7758 val_acc=0.7442 val_f1=0.3059
[Best model saved] exp_outputs/pip2_kl_classifier.pt | best_val_f1=0.3971

[Test Result]
test_loss=0.6383 test_acc=0.7261 test_f1=0.4074

Classification Report:
              precision    recall  f1-score   support

           0     0.9222    0.7924    0.8524       419
           1     0.1574    0.3469    0.2166        49
           2     0.5556    0.5660    0.5607        53
           4     0.0000    0.0000    0.0000         1
    
    accuracy                         0.7261       522
   macro avg     0.4088    0.4263    0.4074       522
weighted avg     0.8114    0.7261    0.7614       522

Confusion Matrix:
[[332  74  13   0]
 [ 22  17  10   0]
 [  6  17  30   0]
 [  0   0   1   0]]

--- Per-joint KL experiment: pip3 ---
Epoch 01 | train_loss=0.7899 train_acc=0.6459 train_f1=0.6311 | val_loss=0.9060 val_acc=0.5504 val_f1=0.3850
Epoch 02 | train_loss=0.5188 train_acc=0.7766 train_f1=0.7671 | val_loss=0.9377 val_acc=0.6221 val_f1=0.4173
Epoch 03 | train_loss=0.4933 train_acc=0.7775 train_f1=0.7791 | val_loss=0.6710 val_acc=0.7229 val_f1=0.3826
Epoch 04 | train_loss=0.4502 train_acc=0.8021 train_f1=0.8016 | val_loss=0.7130 val_acc=0.7054 val_f1=0.3680
Epoch 05 | train_loss=0.4117 train_acc=0.8163 train_f1=0.8157 | val_loss=0.7574 val_acc=0.6647 val_f1=0.4406
Epoch 06 | train_loss=0.3557 train_acc=0.8418 train_f1=0.8414 | val_loss=1.0030 val_acc=0.5736 val_f1=0.4207
Epoch 07 | train_loss=0.3779 train_acc=0.8392 train_f1=0.8404 | val_loss=0.8567 val_acc=0.6105 val_f1=0.3349
Epoch 08 | train_loss=0.3344 train_acc=0.8601 train_f1=0.8562 | val_loss=0.7750 val_acc=0.6686 val_f1=0.4693
Epoch 09 | train_loss=0.3260 train_acc=0.8610 train_f1=0.8625 | val_loss=0.8738 val_acc=0.6647 val_f1=0.4184
Epoch 10 | train_loss=0.2752 train_acc=0.8843 train_f1=0.8808 | val_loss=0.9420 val_acc=0.6182 val_f1=0.3884
Epoch 11 | train_loss=0.2791 train_acc=0.8889 train_f1=0.8905 | val_loss=0.9811 val_acc=0.6143 val_f1=0.3906
Epoch 12 | train_loss=0.2289 train_acc=0.9031 train_f1=0.9031 | val_loss=1.0890 val_acc=0.6105 val_f1=0.3922
[Best model saved] exp_outputs/pip3_kl_classifier.pt | best_val_f1=0.4693

[Test Result]
test_loss=0.7395 test_acc=0.6724 test_f1=0.3516

Classification Report:
              precision    recall  f1-score   support

           0     0.8973    0.7258    0.8025       361
           1     0.2741    0.4684    0.3458        79
           2     0.5618    0.6667    0.6098        75
           3     0.0000    0.0000    0.0000         3
           4     0.0000    0.0000    0.0000         1
    
    accuracy                         0.6724       519
   macro avg     0.3466    0.3722    0.3516       519
weighted avg     0.7470    0.6724    0.6989       519

Confusion Matrix:
[[262  79  20   0   0]
 [ 25  37  17   0   0]
 [  5  19  50   1   0]
 [  0   0   2   0   1]
 [  0   0   0   1   0]]

--- Per-joint KL experiment: pip4 ---
Epoch 01 | train_loss=0.8151 train_acc=0.6484 train_f1=0.6449 | val_loss=1.0639 val_acc=0.4990 val_f1=0.2624
Epoch 02 | train_loss=0.5195 train_acc=0.7706 train_f1=0.7687 | val_loss=1.0845 val_acc=0.4584 val_f1=0.2753
Epoch 03 | train_loss=0.4368 train_acc=0.8033 train_f1=0.8053 | val_loss=0.7580 val_acc=0.6557 val_f1=0.3399
Epoch 04 | train_loss=0.4037 train_acc=0.8157 train_f1=0.8185 | val_loss=0.7333 val_acc=0.6867 val_f1=0.4118
Epoch 05 | train_loss=0.3694 train_acc=0.8327 train_f1=0.8339 | val_loss=0.8098 val_acc=0.6615 val_f1=0.3551
Epoch 06 | train_loss=0.3328 train_acc=0.8584 train_f1=0.8543 | val_loss=0.8514 val_acc=0.6093 val_f1=0.4945
Epoch 07 | train_loss=0.3094 train_acc=0.8712 train_f1=0.8677 | val_loss=0.9334 val_acc=0.5996 val_f1=0.3409
Epoch 08 | train_loss=0.3034 train_acc=0.8687 train_f1=0.8618 | val_loss=0.7758 val_acc=0.6944 val_f1=0.3393
Epoch 09 | train_loss=0.3074 train_acc=0.8675 train_f1=0.8662 | val_loss=0.7619 val_acc=0.6963 val_f1=0.5502
Epoch 10 | train_loss=0.2772 train_acc=0.8807 train_f1=0.8777 | val_loss=0.8667 val_acc=0.6402 val_f1=0.4212
Epoch 11 | train_loss=0.2711 train_acc=0.8919 train_f1=0.8921 | val_loss=0.8415 val_acc=0.6692 val_f1=0.4451
Epoch 12 | train_loss=0.2645 train_acc=0.8911 train_f1=0.8889 | val_loss=1.0281 val_acc=0.5841 val_f1=0.4241
[Best model saved] exp_outputs/pip4_kl_classifier.pt | best_val_f1=0.5502

[Test Result]
test_loss=0.8325 test_acc=0.6447 test_f1=0.3212

Classification Report:
              precision    recall  f1-score   support

           0     0.7737    0.7809    0.7773       324
           1     0.3896    0.4762    0.4286       126
           2     0.5938    0.3016    0.4000        63
           3     0.0000    0.0000    0.0000         1
           4     0.0000    0.0000    0.0000         1
    
    accuracy                         0.6447       515
   macro avg     0.3514    0.3117    0.3212       515
weighted avg     0.6547    0.6447    0.6428       515

Confusion Matrix:
[[253  65   6   0   0]
 [ 60  60   6   0   0]
 [ 14  29  19   1   0]
 [  0   0   1   0   0]
 [  0   0   0   1   0]]

--- Per-joint KL experiment: pip5 ---
Epoch 01 | train_loss=0.8067 train_acc=0.6567 train_f1=0.6539 | val_loss=0.9231 val_acc=0.5631 val_f1=0.4053
Epoch 02 | train_loss=0.5297 train_acc=0.7684 train_f1=0.7610 | val_loss=0.8415 val_acc=0.5806 val_f1=0.4154
Epoch 03 | train_loss=0.4532 train_acc=0.8082 train_f1=0.8066 | val_loss=0.7519 val_acc=0.6505 val_f1=0.2889
Epoch 04 | train_loss=0.4416 train_acc=0.8120 train_f1=0.8084 | val_loss=0.7947 val_acc=0.6485 val_f1=0.3277
Epoch 05 | train_loss=0.3687 train_acc=0.8439 train_f1=0.8370 | val_loss=0.7204 val_acc=0.7049 val_f1=0.3338
Epoch 06 | train_loss=0.3348 train_acc=0.8626 train_f1=0.8664 | val_loss=1.0494 val_acc=0.5146 val_f1=0.2794
Epoch 07 | train_loss=0.2974 train_acc=0.8821 train_f1=0.8806 | val_loss=0.7515 val_acc=0.7146 val_f1=0.3375
Epoch 08 | train_loss=0.2954 train_acc=0.8759 train_f1=0.8743 | val_loss=0.7083 val_acc=0.7437 val_f1=0.4581
Epoch 09 | train_loss=0.2691 train_acc=0.8937 train_f1=0.8962 | val_loss=0.8203 val_acc=0.7126 val_f1=0.3310
Epoch 10 | train_loss=0.2578 train_acc=0.9008 train_f1=0.8971 | val_loss=1.2862 val_acc=0.5107 val_f1=0.2780
Epoch 11 | train_loss=0.2270 train_acc=0.9112 train_f1=0.9072 | val_loss=0.9196 val_acc=0.6680 val_f1=0.3220
Epoch 12 | train_loss=0.2217 train_acc=0.9157 train_f1=0.9157 | val_loss=0.7477 val_acc=0.7709 val_f1=0.3435
[Best model saved] exp_outputs/pip5_kl_classifier.pt | best_val_f1=0.4581

[Test Result]
test_loss=0.6879 test_acc=0.7558 test_f1=0.4809

Classification Report:
              precision    recall  f1-score   support

           0     0.8810    0.8560    0.8683       389
           1     0.2462    0.2424    0.2443        66
           2     0.5634    0.7018    0.6250        57
           3     0.0000    0.0000    0.0000         3
           4     0.5000    1.0000    0.6667         1
    
    accuracy                         0.7558       516
   macro avg     0.4381    0.5600    0.4809       516
weighted avg     0.7588    0.7558    0.7562       516

Confusion Matrix:
[[333  38  18   0   0]
 [ 39  16  11   0   0]
 [  6  10  40   0   1]
 [  0   1   2   0   0]
 [  0   0   0   0   1]]

[Per-joint KL Summary]
  joint  n_train  n_val  n_test  test_acc  test_f1_macro
0  pip2     2413    520     522  0.726054       0.407421
1  pip3     2395    516     519  0.672447       0.351600
2  pip4     2415    517     515  0.644660       0.321167
3  pip5     2409    515     516  0.755814       0.480852

======================================================================
Feature extraction + PCA/UMAP + statistical tests
======================================================================
[Saved] exp_outputs/pca_joint_plot.png
[Saved] exp_outputs/umap_joint_plot.png

[Statistical Test: joint difference on PCA components]
  component  anova_stat        anova_p  kruskal_stat      kruskal_p
0       PC1  285.423531  9.370545e-180    816.426425  1.184906e-176
1       PC2  877.726957   0.000000e+00   2870.929602   0.000000e+00
2       PC3  728.864975   0.000000e+00   2424.489416   0.000000e+00
3       PC4  268.648136  1.842753e-169    846.402825  3.734244e-183
4       PC5  347.568166  1.378332e-217    857.530356  1.441184e-185

======================================================================
Overall Summary
======================================================================
                        experiment  test_acc  test_f1_macro
0             pooled_kl_classifier  0.726834       0.466592
1  pooled_kl_plus_joint_classifier  0.746622       0.586418
2                 joint_classifier  0.976351       0.976348

Interpretation guide:
1. If joint_classifier test_acc is near chance (~0.25), PIP2-5 are hard to distinguish.
2. If pooled_kl_plus_joint only slightly improves over pooled_kl, joint info matters little.
3. If pooled_kl performs as well as or better than per-joint KL models, pooling is justified.
4. If PCA/UMAP shows heavy overlap and stats are weak/non-significant, joint difference is limited.