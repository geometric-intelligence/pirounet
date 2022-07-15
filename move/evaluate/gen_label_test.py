import numpy as np

label_dim = 3
num_gen_per_lab = 10
gen_labels = []
for i in range(label_dim):
    gen_label = i * np.ones(num_gen_per_lab, dtype=int)
    gen_labels = np.concatenate((gen_labels, gen_label))

print(gen_labels)
