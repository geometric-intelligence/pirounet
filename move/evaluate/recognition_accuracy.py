import numpy as np
from sklearn.metrics import multilabel_confusion_matrix, ConfusionMatrixDisplay, confusion_matrix
import matplotlib.pyplot as plt
import matplotlib

filepath="/home/papillon/move/move/data/labels_gen_mathilde_789.csv"
file = open(filepath)

all_labels_mathilde = np.loadtxt(file, delimiter=",")

# Let's start by looking at the overall quality of the dance put out. 

## 1. Determine fraction of dancable material versus non dancable (2nd label is 0)
dance_qual = all_labels_mathilde[:,1]
labels_mathilde = np.delete(all_labels_mathilde, 1, axis=1)

count_dancable = np.count_nonzero(dance_qual==0)/225
print(f'{(1-count_dancable) * 100} % of the dataset is danceable.')

## Of the dancable material, what fraction of it does not respect some physical constraint?
count_no_gravity = np.count_nonzero(dance_qual==2) / (225 - np.count_nonzero(dance_qual==0))
print(f'{(1-count_no_gravity) * 100} % of the dataset respects physical env.')
# for 23% labelled, 21% of the dancable material does not respect some assumed physical constraint
# that LabaNet does not have access to, such as gravity, or uniform ground layer.
# the movement itself respects the constraints of the humna body, and technically
# would be reproducibile.


## Of the dancable material, what fraction of it is jumpy and shows some artificial behaviour?

count_jumpy = np.count_nonzero(dance_qual==3) / (225 - np.count_nonzero(dance_qual==0))
print(f'{(count_jumpy) * 100} % of the dataset is jumpy.')
# for 23% labelled, 25% of the danacble material shows some noticeable artificial effects that are
# not artifacts of the human body. A head getting  large, lack of continuity between
# very first or very last poses.
# We still consider this to be dancable as one can ignore these artifacts and 
# still reproduce the core of the movement. 

#####################################################
# Now we can compare the efforts as labelled by me versus LabaNet
# Only do this for dancable material. Must remove the zero quality first

# Load LabaNet's labels
all_labels_LN = np.load('shuffled_labels_789.npy')

all_labels_LN = np.array(all_labels_LN) + 1.

all_labels_mathilde_no0 = []
labels_LN_no0 = []
for i in range(len(all_labels_mathilde + 1)):
    row_m = all_labels_mathilde[i]
    row_LN = all_labels_LN[i]
    if row_m[1] != 0 and row_m[2] !=0.: # if quality is not 0
        all_labels_mathilde_no0.append(row_m)
        labels_LN_no0.append(row_LN)

labels_ind_m = np.array(all_labels_mathilde_no0)[:,0]
labels_m = np.array(all_labels_mathilde_no0)[:,2]
labels_ln = np.array(labels_LN_no0)

plt.rcParams.update({'font.family':'serif'})
plt.rcParams.update({'font.size':'13'})
conf_mat = confusion_matrix(
    labels_ln,
    labels_m, 
    #normalize = 'true'
    )
classes = ['Low', 'Medium', 'High']
accuracies = conf_mat/conf_mat.sum(1)
fig, ax = plt.subplots(figsize=(4,3))
fig.set_figheight(6)
fig.set_figwidth(7)

cnorm = matplotlib.colors.Normalize(vmin=0, vmax=1)

cb = ax.imshow(accuracies, cmap='Blues', norm=cnorm)
plt.xticks(range(len(classes)), classes, rotation=0)
plt.yticks(range(len(classes)), classes)

for i in range(len(classes)):
    for j in range(len(classes)):
        color='black' if accuracies[j,i] < 0.5 else 'white'
        ax.annotate('{:.2f}'.format(conf_mat[j,i]), (i,j), 
                    color=color, va='center', ha='center')

plt.colorbar(cb, ax=ax, shrink=0.935)
plt.xlabel('Labeler\'s blind classification')
plt.ylabel('Condition given to LabaNet')
plt.title('Labeler versus LabaNet confusion matrix')
purpose = 'self_create'
plt.savefig(fname="confusion/conf_" + str(purpose) + "_789.png", dpi=1200)