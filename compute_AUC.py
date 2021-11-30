from sklearn import metrics

model_accuracy_mean = 0.5679
model_accuracy_std = 0.4953

perturbation_steps = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


# vit_ours neg
perturbed_accuracy_mean = [0.81432, 0.79922, 0.7817, 0.75598, 0.71788, 0.65886, 0.57622, 0.46286, 0.31408, 0.13184, 0] 
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

auc = 0
for i in range(10):
    auc += (perturbed_accuracy_mean[i] + perturbed_accuracy_mean[i+1])/2 * 0.1

perturbation_steps = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

print('AUC: ', auc) # checked: this is equivalent to sklearn's auc
