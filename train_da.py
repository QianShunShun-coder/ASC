import torch
import numpy as np
import torch.nn as nn
import random
import pandas as pd
from torch.utils.data import DataLoader
from cjltest.utils_model import MySGD
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt


class Custom_Dataset(torch.utils.data.dataset.Dataset):
    def __init__(self, _dataset):
        self.dataset = _dataset

    def __getitem__(self, index):
        example, target = self.dataset[index]
        return np.array(example), np.array(target)

    def __len__(self):
        return len(self.dataset)


def list_norm(x):
    # return x
    return [(float(i) - min(x)) / (max(x) - min(x)) for i in x]


def read_data(path, noise_flag):
    df = pd.read_excel(path)

    feature_list = []
    feature_list.append(list_norm(df["吸烟指数"].tolist()))
    feature_list.append(list_norm(df["性别"].tolist()))
    feature_list.append(list_norm(df["年龄"].tolist()))
    feature_list.append(list_norm(df["吸烟分级"].tolist()))
    feature_list.append(list_norm(df["肿瘤位置"].tolist()))
    feature_list.append(list_norm(df["CEA"].tolist()))
    feature_list.append(list_norm(df["NSE"].tolist()))
    feature_list.append(list_norm(df["T分期"].tolist()))
    feature_list.append(list_norm(df["N分期"].tolist()))
    feature_list.append(list_norm(df["M分期"].tolist()))
    feature_list.append(list_norm(df["ECOG评分"].tolist()))
    feature_list.append(list_norm(df["SII(血小板*中性粒/淋巴)"].tolist()))
    feature_list.append(list_norm(df["NLR(中性粒/淋巴）"].tolist()))
    feature_list.append(list_norm(df["PLR(血小板/淋巴）"].tolist()))
    feature_list.append(list_norm(df["LMR（淋巴/单核）"].tolist()))
    # for features in feature_list:
    #     print(len(features))
    print(len(feature_list))
    train_list = []

    for i in range(0, len(feature_list[0])):
        train_subject = []
        target_subject = []
        feature_idx = 0
        for features in feature_list:
            target_subject.append(features[i])
            if noise_flag and random.random() < 0.2:
                train_subject.append(0.0)
            else:
                train_subject.append(features[i])
            # feature_idx += 1

        train_list.append((train_subject, target_subject))
        # print(train_list[-1])
    data_loader = DataLoader(dataset=Custom_Dataset(train_list), batch_size=8, shuffle=True)
    return data_loader


if __name__ == '__main__':
    train_path = 'data/train.xlsx'
    test_path = 'data/test.xlsx'

    train_loader = read_data(train_path, 1)
    test_loader = read_data(test_path, 1)

    print("Load Success")

    import autoencoder

    global_model = autoencoder.denoising_model()
    w_optimizer = MySGD(global_model.parameters(), lr=0.01, weight_decay=1e-5)
    criterion = nn.MSELoss()

    y_true = []
    y_score = []

    for epoch in range(0, 500):
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
            # print(inputs)
            targets = targets.float()
            output = global_model(inputs)
            # if epoch > 900:
            #     print("input", inputs[0])
            #     print("output", output[0])
            #     print("target", targets[0])
            loss = criterion(output, targets)
            # print(output)
            loss.backward()

            w_optimizer.step()

            # _, predicted = output.max(1)
            # total += targets.size(0)
            # correct += predicted.eq(targets).sum().item()
            average_loss += loss / E

            # break
        # print("Training epoch:", epoch, 'loss: ', average_loss)

        # total = 0
        # correct = 0
        # batch_idx = 0

        average_loss = 0
        global_model.eval()
        for inputs, targets in test_loader:

            inputs = inputs.float()
            targets = targets.float()
            w_optimizer.zero_grad()
            output = global_model(inputs.float())
            if epoch > 450:
                print("input", inputs[0])
                print("output", output[0])
                print("target", targets[0])

            loss = criterion(output, targets)

            # _, predicted = output.max(1)
            # total += targets.size(0)
            # correct += predicted.eq(targets).sum().item()
            average_loss += loss / E

            # Store predicted and true labels for ROC curve
            y_true += targets.tolist()
            y_score += output.tolist()

        print("Test epoch:", epoch, 'loss: ', average_loss)

    # Compute ROC curve and AUC
    y_true = np.array(y_true)
    y_score = np.array(y_score)

    ```python
    import torch
    import numpy as np
    import torch.nn as nn
    import random
    import pandas as pd
    from torch.utils.data import DataLoader
    from cjltest.utils_model import MySGD
    from sklearn.metrics import roc_curve, auc
    import matplotlib.pyplot as plt


    class Custom_Dataset(torch.utils.data.dataset.Dataset):
        def __init__(self, _dataset):
            self.dataset = _dataset

        def __getitem__(self, index):
            example, target = self.dataset[index]
            return np.array(example), np.array(target)

        def __len__(self):
            return len(self.dataset)


    def list_norm(x):
        # return x
        return [(float(i) - min(x)) / (max(x) - min(x)) for i in x]


    def read_data(path, noise_flag):
        df = pd.read_excel(path)

        feature_list = []
        feature_list.append(list_norm(df["吸烟指数"].tolist()))
        feature_list.append(list_norm(df["性别"].tolist()))
        feature_list.append(list_norm(df["年龄"].tolist()))
        feature_list.append(list_norm(df["吸烟分级"].tolist()))
        feature_list.append(list_norm(df["肿瘤位置"].tolist()))
        feature_list.append(list_norm(df["CEA"].tolist()))
        feature_list.append(list_norm(df["NSE"].tolist()))
        feature_list.append(list_norm(df["T分期"].tolist()))
        feature_list.append(list_norm(df["N分期"].tolist()))
        feature_list.append(list_norm(df["M分期"].tolist()))
        feature_list.append(list_norm(df["ECOG评分"].tolist()))
        feature_list.append(list_norm(df["SII(血小板*中性粒/淋巴)"].tolist()))
        feature_list.append(list_norm(df["NLR(中性粒/淋巴）"].tolist()))
        feature_list.append(list_norm(df["PLR(血小板/淋巴）"].tolist()))
        feature_list.append(list_norm(df["LMR（淋巴/单核）"].tolist()))
        # for features in feature_list:
        #     print(len(features))
        print(len(feature_list))
        train_list = []

        for i in range(0, len(feature_list[0])):
            train_subject = []
            target_subject = []
            feature_idx = 0
            for features in feature_list:
                target_subject.append(features[i])
                if noise_flag and random.random() < 0.2:
                    train_subject.append(0.0)
                else:
                    train_subject.append(features[i])
                # feature_idx += 1

            train_list.append((train_subject, target_subject))
            # print(train_list[-1])
        data_loader = DataLoader(dataset=Custom_Dataset(train_list), batch_size=8, shuffle=True)
        return data_loader


    if __name__ == '__main__':
        train_path = 'data/train.xlsx'
        test_path = 'data/test.xlsx'

        train_loader = read_data(train_path, 1)
        test_loader = read_data(test_path, 1)

        print("Load Success")

        import autoencoder

        global_model = autoencoder.denoising_model()
        w_optimizer = MySGD(global_model.parameters(), lr=0.01, weight_decay=1e-5)
        criterion = nn.MSELoss()

        y_true = []
        y_score = []

        for epoch in range(0, 500):
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
                # print(inputs)
                targets = targets.float()
                output = global_model(inputs)
                # if epoch > 900:
                #     print("input", inputs[0])
                #     print("output", output[0])
                #     print("target", targets[0])
                loss = criterion(output, targets)
                # print(output)
                loss.backward()

                w_optimizer.step()

                # _, predicted = output.max(1)
                # total += targets.size(0)
                # correct += predicted.eq(targets).sum().item()
                average_loss += loss / E

                # break
            # print("Training epoch:", epoch, 'loss: ', average_loss)

            # total = 0
            # correct = 0
            # batch_idx = 0

            average_loss = 0
            global_model.eval()
            for inputs, targets in test_loader:

                inputs = inputs.float()
                targets = targets.float()
                w_optimizer.zero_grad()
                output = global_model(inputs.float())
                if epoch > 450:
                    print("input", inputs[0])
                    print("output", output[0])
                    print("target", targets[0])

                loss = criterion(output, targets)

                # _, predicted = output.max(1)
                # total += targets.size(0)
                # correct += predicted.eq(targets).sum().item()
                average_loss += loss / E

                # Store predicted and true labels for ROC curve
                y_true += targets.tolist()
                y_score += output.tolist()

            print("Test epoch:", epoch, 'loss: ', average_loss)

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

validation_loader = read_data("data/validation.xlsx", noise_flag=1)

y_true = []
y_score = []

average_loss = 0
global_model.eval()
for inputs, targets in validation_loader:
    inputs = inputs.float()
    targets = targets.float()
    w_optimizer.zero_grad()
    output = global_model(inputs.float())
    loss = criterion(output, targets)

    average_loss += loss / len(validation_loader)

    # Store predicted and true labels for ROC curve
    y_true += targets.tolist()
    y_score += output.tolist()

print("Validation loss:", average_loss)

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
