
model_accuracy_mean = 0.5679
model_accuracy_std = 0.4953

perturbation_steps = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
perturbed_accuracy_mean = [0.52274, 0.49256, 0.45962, 0.4166,  0.36808, 0.30808, 0.232, 0.14682, 0.0552 ]
perturbed_accuracy_mean = [0.5679, 0.52274, 0.49256, 0.45962, 0.4166,  0.36808, 0.30808, 0.232, 0.14682, 0.0552, 0] 

perturbed_accuracy_std = [0.49948262, 0.49994464, 0.49836679, 0.49299538, 0.48228323, 0.46169981, 0.42210899, 0.35392639, 0.22837023]

perturbed_accuracy_mean = [0.80314, 0.78168, 0.76424, 0.73972, 0.7043,  0.65024, 0.57338, 0.45624, 0.29448, 0.1045, 0 ]
auc = 0
for i in range(8):
    auc += (perturbed_accuracy_mean[i] + perturbed_accuracy_mean[i+1])/2 * 0.1

print(auc)
print(auc/ model_accuracy_mean)