from sklearn import metrics
import numpy as np
model_accuracy_mean = 0.5679
model_accuracy_std = 0.4953

perturbation_steps = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


# vit_ours neg
# perturbed_accuracy_mean = [0.81432, 0.79922, 0.7817, 0.75598, 0.71788, 0.65886, 0.57622, 0.46286, 0.31408, 0.13184, 0]
# vit_ours neg orginal predictions
# perturbed_accuracy_mean = [1, 0.91374, 0.881, 0.84296, 0.7943, 0.72446, 0.6294, 0.50392, 0.34044, 0.14204, 0]


# vit_ours pos
# perturbed_accuracy_mean = [0.81432, 0.63016, 0.47856, 0.34242, 0.22296, 0.13578, 0.0724, 0.03388, 0.01356, 0.00464, 0]
# vit_ours pos orginal predictions
perturbed_accuracy_mean = [1, 0.6599, 0.4882, 0.3441,
                           0.22202, 0.1329, 0.07042, 0.03296, 0.0132, 0.0044, 0]


# vit_ours_NCC pos
perturbed_accuracy_mean = [0.76394, 0.42608, 0.25682, 0.14892,
                           0.0808,  0.04192, 0.02054, 0.01002, 0.00424, 0.00276, 0]
# vit_ours pos orginal predictions
# perturbed_accuracy_mean = [1, 0.44858, 0.26092, 0.14928,
#                            0.07912, 0.04036, 0.0192,  0.00972, 0.00422, 0.00282, 0]

# vit_ours 1/3 neg
#perturbed_accuracy_mean = [0.81432, 0.8, 0.78074, 0.75592, 0.71802, 0.66016, 0.57564, 0.46252, 0.31462, 0.13328, 0]
# vit_ours 1/3 neg original prediction
#perturbed_accuracy_mean = [1, 0.914221, 0.88106, 0.84278, 0.79354, 0.72622, 0.63026, 0.50296, 0.3411, 0.14348, 0]

# vit_ours 1/3 pos
#perturbed_accuracy_mean = [0.81432, 0.63082, 0.47878, 0.34118, 0.2236, 0.1346, 0.07262, 0.03336, 0.01374, 0.00476, 0]
# vit_ours 1/3 pos original prediction
#perturbed_accuracy_mean = [1, 0.65944, 0.48794, 0.34304, 0.22218, 0.1326, 0.071, 0.03274, 0.0135, 0.00456, 0]

# vit ours LRP-eps
#perturbed_accuracy_mean = [0.81432, 0.79928, 0.78076, 0.75648, 0.7162, 0.65838, 0.5744, 0.4627, 0.31252, 0.1321, 0]
#perturbed_accuracy_mean = [1, 0.91252, 0.87954, 0.8424, 0.7928, 0.72442, 0.62758, 0.50352, 0.33948, 0.14238, 0]

###### BASELINES ######
# Rollout - vit ours neg
#perturbed_accuracy_mean = [0.81432, 0.79556, 0.77424, 0.74198, 0.69568, 0.63094, 0.54494, 0.43042, 0.29146, 0.12636, 0]
# Rollout - vit ours neg original prediction
#perturbed_accuracy_mean = [1, 0.90378, 0.86096, 0.81542, 0.75604, 0.67808, 0.58048, 0.4556,  0.30642, 0.13298, 0]

# Rollout - vit ours pos
# perturbed_accuracy_mean = [0.81432, 0.65148, 0.50906, 0.374, 0.25924, 0.16112, 0.0907, 0.0437, 0.01802, 0.00576, 0]
# Rollout - vit ours pos original prediction
# perturbed_accuracy_mean = [1, 0.70446, 0.53758, 0.3931, 0.27006, 0.1668, 0.09368, 0.04486, 0.01842, 0.0054, 0]

# LRP - vit ours neg
# perturbed_accuracy_mean = [0.81432, 0.76928, 0.73244, 0.69184, 0.63948, 0.56702, 0.46976, 0.3446, 0.19628, 0.05924, 0]
# LRP - vit ours neg original prediction
# perturbed_accuracy_mean = [1, 0.84746, 0.79418, 0.743, 0.68054, 0.60078, 0.49284, 0.3619, 0.20548, 0.06176, 0]

# LRP - vit ours pos
# perturbed_accuracy_mean = [0.81432, 0.7514, 0.67606, 0.59874, 0.51452, 0.42524, 0.33736, 0.23904, 0.13444, 0.03706, 0]
# LRP - vit ours pos original prediction
# perturbed_accuracy_mean = [1, 0.82602, 0.73088, 0.6406, 0.54534, 0.44804, 0.35364, 0.2508, 0.13982, 0.03866, 0]

# Partial LRP - vit ours neg
# perturbed_accuracy_mean = [0.81432, 0.7931, 0.7673, 0.7283, 0.67602, 0.59924, 0.50492, 0.38456, 0.24858, 0.10036, 0]
# Partial LRP - vit ours neg original prediction
# perturbed_accuracy_mean = [1, 0.89822, 0.84988, 0.7959, 0.7322, 0.64416, 0.53804, 0.40634, 0.26194, 0.10446, 0]

# Partial LRP - vit ours pos
# perturbed_accuracy_mean = [0.81432, 0.664, 0.53254, 0.40498, 0.28812, 0.18988, 0.11472, 0.05818, 0.02438, 0.00718, 0]
# Partial LRP - vit ours pos original prediction
# perturbed_accuracy_mean = [1, 0.71984, 0.56588, 0.42516, 0.30162, 0.19758, 0.11854, 0.05958, 0.02428, 0.00728, 0]


# GradCAM - neg
# perturbed_accuracy_mean = [0.81432, 0.79346, 0.74994, 0.6807, 0.58306, 0.45922, 0.33518, 0.2148, 0.11436, 0.03876,0]
# GradCAM - neg, original prediction
# perturbed_accuracy_mean = [1, 0.90234, 0.84016, 0.75304, 0.63902, 0.50162, 0.36484, 0.2343, 0.12416, 0.04214, 0]
# GradCAM - pos
# perturbed_accuracy_mean = [0.81432, 0.72076, 0.64598, 0.5715, 0.49384, 0.41376, 0.32942, 0.24482, 0.14402, 0.04708, 0]
# GradCAM - pos, original prediction
#perturbed_accuracy_mean = [1, 0.78424, 0.6872, 0.59992, 0.51422, 0.42752, 0.33808, 0.24938, 0.14644, 0.04784, 0]

# Softmax - neg
#perturbed_accuracy_mean = [0.81432, 0.79992, 0.78184, 0.75678, 0.71776, 0.65806, 0.57496, 0.46228, 0.31384, 0.13194, 0]
# Softmax - neg, original prediction
#perturbed_accuracy_mean = [1, 0.91522, 0.88138, 0.84368, 0.79452, 0.72494, 0.62848, 0.504, 0.34066, 0.1423, 0]
# Softmax - pos
#perturbed_accuracy_mean = [0.81432, 0.63202, 0.4789, 0.34214, 0.22368, 0.13498, 0.07242, 0.03324, 0.01312, 0.00462, 0]
# Softmax - pos, original prediction
# perturbed_accuracy_mean = [1, 0.66004, 0.48768, 0.34362,
#                            0.22278, 0.13228, 0.07114, 0.0322, 0.0127, 0.00452, 0]


###### ABLATION STUDY ######
# 1001 images, "last_layer". AUC: 0.6905
#perturbed_accuracy_mean = [1, 0.95504496, 0.93106893, 0.88911089, 0.85314685, 0.79120879, 0.74025974, 0.62337662, 0.44155844, 0.18081918, 0]
# 1001 images, "second_layer". AUC: 0.5561
#perturbed_accuracy_mean = [1, 0.94405594, 0.89010989, 0.83416583, 0.74925075, 0.62137862, 0.47252747, 0.32467532, 0.17682318, 0.04895105, 0]
# 1001 images, "first_layer". AUC:  0.4974
#perturbed_accuracy_mean = [1, 0.93106893, 0.86213786, 0.77722278, 0.64835165, 0.5014985,  0.36563437, 0.23676324, 0.11988012, 0.03196803, 0]
# 1001 images, "transformer_attribution". neg. AUC: 0.6938
#perturbed_accuracy_mean = [0.92207, 0.9040959, 0.88311688, 0.85814186, 0.83616384, 0.79120879, 0.74125874, 0.66933067, 0.52947053, 0.26473526, 0]
# 1001 images, "transformer_attribution", neg, original prediction. Checked that it is not random (got the exact same twice) AUC:   0.7271
#perturbed_accuracy_mean = [1, 0.94905095, 0.92807193, 0.8961039,  0.87912088, 0.82617383, 0.77622378, 0.6973027,  0.54545455, 0.27372627, 0]

# #### SOFTMAX ####
# perturbed_accuracy_mean = [0.92207, 0.90609391, 0.88011988, 0.85614386, 0.82517483, 0.79320679, 0.73526474, 0.67032967, 0.52647353, 0.27372627, 0]
# perturbed_accuracy_mean = [1, 0.95604396, 0.92807193, 0.8981019, 0.87012987, 0.82817183, 0.76723277, 0.7032967,  0.54345654, 0.28271728, 0]


# 1001 images, "transformer_attribution". pos. AUC: 0.2498
#perturbed_accuracy_mean = [0.9220, 0.71628372, 0.53346653, 0.35264735, 0.21978022, 0.11688312, 0.06393606, 0.02597403, 0.00899101, 0, 0]
# 1001 images, "transformer_attribution", pos, original prediction. AUC: 0.2552
#perturbed_accuracy_mean = [1, 0.72927073, 0.53746254, 0.35164835, 0.21878122, 0.11688312, 0.06193806, 0.02697303, 0.00999001, 0, 0]

# 1001 images, "lrp". AUC:    0.6054
#perturbed_accuracy_mean = [1, 0.92307692, 0.88211788, 0.84815185, 0.79220779, 0.70629371, 0.60939061, 0.45854146, 0.27372627, 0.06093906, 0]
# 1001 images, "rollout". AUC:   0.7131
#perturbed_accuracy_mean = [1, 0.94605395, 0.91908092, 0.88911089, 0.84415584, 0.80919081, 0.75624376, 0.65734266, 0.51548452, 0.29470529,0]

#### RISE ####
# RISE, 1001 images, neg. AUC: 0.7002
# perturbed_accuracy_mean = [0.92205115, 0.91008991, 0.9000999, 0.89010989, 0.87412587, 0.82717283, 0.75924076, 0.63936064, 0.47552448, 0.26573427, 0]
# RISE, 1001 images, neg, original prediction. AUC: 0.7476
# perturbed_accuracy_mean = [1, 0.97702298, 0.96403596, 0.95304695, 0.93006993, 0.88411589, 0.81018981, 0.68531469, 0.4995005, 0.27272728, 0]
# RISE, 1001 images, pos. AUC: 0.3511
#perturbed_accuracy_mean = (np.array([319, 242, 190, 163, 124,  93, 70,  39,  13,   4, 0])  + np.array([604, 476, 397, 344, 292, 245, 171, 106,  60,  25, 0]))/1001
# RISE, 1001 images, pos, original prediction. AUC: 0.3471
#perturbed_accuracy_mean = (np.array([332, 244, 187, 162, 122,  92,  68,  38,  12,   3, 0]) + np.array([669, 480, 393, 334, 276, 227, 161,  99,  55,  21, 0]))/1001

auc = 0
for i in range(10):
    auc += (perturbed_accuracy_mean[i] + perturbed_accuracy_mean[i+1])/2 * 0.1

perturbation_steps = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

print('AUC: ', auc)  # checked: this is equivalent to sklearn's auc
