import csv

import numpy as np

filepath = "/home/papillon/move/move/data/labels.csv"

file = open(filepath)
labels_with_index = np.loadtxt(file, delimiter=",")
labels = np.delete(labels_with_index, 0, axis=1)
print(labels.shape)

amount_one = list(labels[:, 1]).count(1)
amount_two = list(labels[:, 1]).count(2)
amount_three = list(labels[:, 1]).count(3)
amount_four = list(labels[:, 1]).count(4)
print(amount_one)
print(amount_two)
print(amount_three)
print(amount_four)
