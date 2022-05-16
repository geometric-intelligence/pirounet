import csv

import numpy as np

# filepath = "/home/papillon/move/move/data/labels.csv"

# file = open(filepath)
# labels_with_index = np.loadtxt(file, delimiter=",")
# file.close()

# new_lab_total = np.zeros((1,3))
# similarities = 0
# for j in range(len(labels_with_index)):
#     if j == 0:
#         continue

#     if j != 0:
#         index = labels_with_index[j][0]
#         index_b = labels_with_index[j-1][0]
#         seq_len = 40 #index - index_b - 1
#         new_lab = np.zeros((int(seq_len), 3))

#         space = labels_with_index[j][1]
#         time = labels_with_index[j][2]
#         bspace = labels_with_index[j-1][1]
#         btime = labels_with_index[j-1][2]
        

#         if space == bspace and time == btime:
#             similarities += 1
#             for i in range(int(seq_len)):
#                 new_lab[i][0] = int(index + i + 1)
#                 new_lab[i][1] = int(space)
#                 new_lab[i][2] = int(time)
#                 new_lab = np.array(new_lab).reshape((-1,3))

#             new_lab_total = np.append(new_lab_total, new_lab, axis=0)

#         else:
#             continue


# new_lab_total = np.delete(new_lab_total, 0, 0)

# with open('smart_labels.csv', 'w', newline='') as file:
#     writer = csv.writer(file, delimiter=',')
#     writer.writerows(new_lab_total)
# file.close()

# extra_frames = 5
# augmented_labels = []
# for j in range(len(labels_with_index)):
#     index_labelled = labels_with_index[j][0]
#     space = labels_with_index[j][1]
#     time = labels_with_index[j][2]
#     to_add = []

#     if index_labelled == 0:
#         for i in range(extra_frames + 2):
#             extra_label = [i, space, time]
#             to_add = np.append(to_add, extra_label, axis=0)

#         augmented_labels = np.append(augmented_labels, to_add, axis=0)
    
#     if index_labelled != 0:
#         for i in range(extra_frames + 1):
#             i_rev = extra_frames - i
#             extra_label_neg = [index_labelled - (i_rev + 1), space, time]
#             to_add = np.append(to_add, extra_label_neg, axis=0)

#         to_add = np.append(to_add, list(labels_with_index[j]), axis=0)

#         for i in range(extra_frames + 1):        
#             extra_label_pos = [index_labelled + (i + 1), space, time] 
#             to_add = np.append(to_add, extra_label_pos, axis=0)       

#         augmented_labels = np.append(augmented_labels, to_add, axis=0)

# augmented_labels = augmented_labels.reshape(int(len(augmented_labels) / 3.), 3)

# with open('aug_labels.csv', 'w', newline='') as file:
#     writer = csv.writer(file, delimiter=',')
#     writer.writerows(augmented_labels)

smafilepath = "/home/papillon/move/move/data/smart_labels.csv"
augfilepath = "/home/papillon/move/move/data/aug_labels.csv"

file = open(smafilepath)
smart_labels = np.loadtxt(file, delimiter=",")
file.close()

file = open(augfilepath)
aug_labels = np.loadtxt(file, delimiter=",")
file.close()

labels = np.append(smart_labels, aug_labels, axis=0)

uniquelabels = labels[np.unique(labels[:,0], axis=0, return_index=True)[1]]

with open('labels.csv', 'w', newline='') as file:
    writer = csv.writer(file, delimiter=',')
    writer.writerows(uniquelabels)


# ulabels = np.unique(labels, axis = 0)

# ilabels = (np.delete(np.delete(ulabels, 2, axis=1), 1, axis=1))
# ilabels = ilabels.reshape((len(ilabels)))

# print(ilabels.shape)
# print(ilabels)
# u, counts = np.unique(ilabels,return_counts=True)