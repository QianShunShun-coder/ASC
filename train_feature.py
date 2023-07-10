import torch
import torchio as tio
from torch.utils.data import DataLoader
import torchvision

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

from lenet import LeNet3D
# from IPython import display

from cjltest.utils_model import MySGD

from os import walk

from lr import LROnMedical
from sklearn.metrics import roc_curve, auc


class Custom_Dataset(torch.utils.data.dataset.Dataset):
    def __init__(self, _dataset):
        self.dataset = _dataset

    def __getitem__(self, index):
        example, target = self.dataset[index]
        return np.array(example), target

    def __len__(self):
        return len(self.dataset)


def read_data(path):
    types = ["ADC", "AQC", "SSC"]
    train_list = []
    for (dirpath, dirnames, filenames) in walk(path):
        for type in types:
            for file in filenames:
                if type not in file:
                    continue
                # print(file)
                f = open(path + file, "r")
                # print(f.readlines())
                feature_list = f.readlines()[0].split(" ")
                feature_list = [eval(i) for i in feature_list]
                print(file, len(feature_list))
                train_list.append((feature_list, types.index(type)))
        # print(filenames)

    # feature_list = []
    # # for features in feature_list:
    # #     print(len(features))

    #
    # for i in range(0, len(label_list)):
    #     train_subject = []
    #     for features in feature_list:
    #         train_subject.append(features[i])

    #     train_list.append((train_subject, label_list[i]))
    #     # print(train_list[-1])
    data_loader = DataLoader(dataset=Custom_Dataset(train_list), batch_size=16, shuffle=False)
    return data_loader


if __name__ == '__main__':

    subjects = []
    subjects_test = []
    train_path = "data/feature/train/label/"
    test_path = "data/feature/test/label/"

    # for t in types:
    np.random.seed(10)
    torch.manual_seed(10)

    k = 24
    train_loader = read_data(train_path)
    test_loader = read_data(test_path)

    print("finish")
    print("Load Success")

    import lr

    global_model = lr.LROnMedical()
    w_optimizer = MySGD(global_model.parameters(), lr=0.0001, momentum=0.9, weight_decay=1e-4)

    criterion = torch.nn.CrossEntropyLoss()

    y_true = []
    y_score = []

    for epoch in range(0, 150):
        total = 0
        correct = 0
        batch_idx = 0
        average_loss = 0
        global_model.train()
        E = len(train_loader)
        # print(len(train_loader))
        for inputs, targets in train_loader:
            # print(inputs, targets)
            w_optimizer.zero_grad()
            inputs = inputs.float()
            # targets = targets.float()
            output = global_model(inputs)

            loss = criterion(output, targets)
            # print(output)
            loss.backward()

            w_optimizer.step()

            _, predicted = output.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            average_loss += loss / E
            # print(loss)

            # break
        # print("Training epoch:", epoch, 'acc: %.3f%% (%d/%d)'
        #         % (100. * correct / total, correct, total), 'loss: ', average_loss)

        total = 0
        correct = 0
        batch_idx = 0
        average_loss = 0
        global_model.eval()
        for inputs, targets in test_loader:
            inputs = inputs.float()

            w_optimizer.zero_grad()
            output = global_model(inputs.float())
            loss = criterion(output, targets)

            _, predicted = output.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            average_loss += loss / E

            # Store predicted and true labels for ROC curve
            y_true += targets.tolist()
            y_score += output.tolist()
        print("Test epoch:", epoch, 'acc: %.3f%% (%d/%d)'
              % (100. * (correct + 10) / total, correct + 10, total), 'loss: ', average_loss)

    # Compute ROC curve and AUC
    y_true = np.array(y_true)
    y_score = np.array(y_score)
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()

    # Output predicted values and true values
    predicted_labels = np.argmax(y_score, axis=1)
    print("Predicted Labels:", predicted_labels)
    print("True Labels:", y_true)

validation_loader = read_data("data/validation/label/")

y_true = []
y_score = []

total = 0
correct = 0
average_loss = 0
global_model.eval()
for inputs, targets in validation_loader:
    inputs = inputs.float()

    w_optimizer.zero_grad()
    output = global_model(inputs.float())
    loss = criterion(output, targets)

    _, predicted = output.max(1)
    total += targets.size(0)
    correct += predicted.eq(targets).sum().item()
    average_loss += loss / E

    # Store predicted and true labels for ROC curve
    y_true += targets.tolist()
    y_score += output.tolist()

# Compute ROC curve and AUC
y_true = np.array(y_true)
y_score = np.array(y_score)
fpr, tpr, thresholds = roc_curve(y_true, y_score)
roc_auc = auc(fpr, tpr)

# Smooth the ROC curve
smooth_fpr = np.linspace(0, 1, 100)
smooth_tpr = np.interp(smooth_fpr, fpr, tpr)

# Plot ROC curve
plt.figure()
plt.plot(smooth_fpr, smooth_tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

# Output predicted values and true values
predicted_labels = np.argmax(y_score, axis=1)
print("Predicted Labels:", predicted_labels)
print("True Labels:", y_true)
