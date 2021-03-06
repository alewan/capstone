# Written by Aleksei Wan on 14.01.2020
# A short script to plot the accuracy from the CSV generated by the audio nn (at location ~/Downloads/acc.csv)

import numpy as np
import matplotlib.pyplot as plt
from os import path
import seaborn as sns

# Use seaborn so it's prettier
sns.set()
sns.set_context('paper')

NUMBER_OF_EPOCHS = 500
BATCHES_PER_EPOCH = int(55 * (100 / NUMBER_OF_EPOCHS))  # or use `None`
CSV_PATH = path.join(path.expanduser('~'), 'Downloads', 'acc.csv')

my_data = np.genfromtxt(CSV_PATH, delimiter=',')

# If parameter is specified, average the accuracy to make it per epoch instead of per batch
if BATCHES_PER_EPOCH is not None:
    s = 0.0
    holder = np.zeros(int(np.size(my_data)/BATCHES_PER_EPOCH) + 1)
    for idx, val in enumerate(my_data):
        holder[int(idx / BATCHES_PER_EPOCH)] += val

    holder /= BATCHES_PER_EPOCH
    holder[-1] *= BATCHES_PER_EPOCH / (np.size(my_data) % BATCHES_PER_EPOCH)  # Correct the last epoch

    plt.plot(holder)
    plt.xlabel('Epoch Number')
    print('Max value:', np.argmax(holder))
else:
    plt.plot(my_data)
    plt.xlabel('Iterations')
    print('Max value:', np.argmax(my_data))

plt.title('Accuracy for Provided Data Set')
plt.ylabel('Accuracy')
plt.show()
