from sklearn import metrics

model_accuracy_mean = 0.5679
model_accuracy_std = 0.4953

perturbation_steps = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


# vit_ours neg
# perturbed_accuracy_mean = [0.81432, 0.79922, 0.7817, 0.75598, 0.71788, 0.65886, 0.57622, 0.46286, 0.31408, 0.13184, 0] 
# vit_ours neg orginal predictions
#perturbed_accuracy_mean = [1, 0.91374, 0.881, 0.84296, 0.7943, 0.72446, 0.6294, 0.50392, 0.34044, 0.14204, 0] 

# vit_ours 1/3 neg
#perturbed_accuracy_mean = [0.81432, 0.8, 0.78074, 0.75592, 0.71802, 0.66016, 0.57564, 0.46252, 0.31462, 0.13328, 0] 
# vit_ours 1/3 neg original prediction
#perturbed_accuracy_mean = [1, 0.914221, 0.88106, 0.84278, 0.79354, 0.72622, 0.63026, 0.50296, 0.3411, 0.14348, 0]

# vit_ours 1/3 pos
#perturbed_accuracy_mean = [0.81432, 0.63082, 0.47878, 0.34118, 0.2236, 0.1346, 0.07262, 0.03336, 0.01374, 0.00476, 0]
# vit_ours 1/3 pos original prediction
#perturbed_accuracy_mean = [1, 0.65944, 0.48794, 0.34304, 0.22218, 0.1326, 0.071, 0.03274, 0.0135, 0.00456, 0]

# vit ours LRP-eps 
perturbed_accuracy_mean = [0.81432, 0.79928, 0.78076, 0.75648, 0.7162, 0.65838, 0.5744, 0.4627, 0.31252, 0.1321, 0] 
#perturbed_accuracy_mean = [1, 0.91252, 0.87954, 0.8424, 0.7928, 0.72442, 0.62758, 0.50352, 0.33948, 0.14238, 0]



###### ABLATION STUDY ######
# 1000 images, "last_layer". AUC: 0.6905
perturbed_accuracy_mean = [1, 0.95504496, 0.93106893, 0.88911089, 0.85314685, 0.79120879, 0.74025974, 0.62337662, 0.44155844, 0.18081918, 0]
# 1000 images, "second_layer". AUC: 0.5561
perturbed_accuracy_mean = [1, 0.94405594, 0.89010989, 0.83416583, 0.74925075, 0.62137862, 0.47252747, 0.32467532, 0.17682318, 0.04895105, 0]
# 1000 images, "first_layer". AUC:  0.4974
perturbed_accuracy_mean = [1, 0.93106893, 0.86213786, 0.77722278, 0.64835165, 0.5014985,  0.36563437, 0.23676324, 0.11988012, 0.03196803, 0]
# 1000 images, "transformer_attribution". AUC:   0.7271
perturbed_accuracy_mean = [1, 0.94905095, 0.92807193, 0.8961039,  0.87912088, 0.82617383, 0.77622378, 0.6973027,  0.54545455, 0.27372627, 0]
# 1000 images, "partial_lrp". AUC:    0.6054
perturbed_accuracy_mean = [1, 0.92307692, 0.88211788, 0.84815185, 0.79220779, 0.70629371, 0.60939061, 0.45854146, 0.27372627, 0.06093906, 0]
# 1000 images, "rollout". AUC:   0.7131
perturbed_accuracy_mean = [1, 0.94605395, 0.91908092, 0.88911089, 0.84415584, 0.80919081, 0.75624376, 0.65734266, 0.51548452, 0.29470529,0]

auc = 0
for i in range(10):
    auc += (perturbed_accuracy_mean[i] + perturbed_accuracy_mean[i+1])/2 * 0.1

perturbation_steps = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

print('AUC: ', auc) # checked: this is equivalent to sklearn's auc
