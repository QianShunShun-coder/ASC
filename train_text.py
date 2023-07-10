import torch
import numpy as np
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
        return np.array(example), target

    def __len__(self):
        return len(self.dataset)


def label_to_num(label_list, types_chinese):
    result_list = []
    for label in label_list:
        result_list.append(types_chinese.index(label))
    return result_list


def position_to_num(position_list):
    result_list = []
    for position in position_list:
        if "左" in position:
            result_list.append(0)
        else:
            result_list.append(1)
    return result_list


def list_norm(x):
    return [(float(i) - min(x)) / (max(x) - min(x)) for i in x]


def read_data(path):
    types_chinese = ["腺癌", "腺鳞癌", "鳞癌"]
    df = pd.read_excel(path)
    label_list = df["病理类型"].tolist()
    label_list = label_to_num(label_list, types_chinese)

    feature_list = []
    position_feature = position_to_num(df["肿瘤位置（肺段）"].tolist())
    feature_list.append(list_norm(position_feature))
    feature_list.append(list_norm(df["吸烟指数"].tolist()))
    feature_list.append(list_norm(df["性别"].tolist()))
    feature_list.append(list_norm(df["年龄"].tolist()))
    feature_list.append(list_norm(df["吸烟分级"].tolist()))
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

    train_list = []
    for i in range(len(label_list)):
        train_subject = []
        for features in feature_list:
            train_subject.append(features[i])
        train_list.append((train_subject, label_list[i]))

    data_loader = DataLoader(dataset=Custom_Dataset(train_list), batch_size=16, shuffle=False)
    return data_loader


if __name__ == '__main__':
    types = ["ADC", "AQC", "SSC"]
    train_path = 'data/train.xlsx'
    test_path = 'data/test.xlsx'
    np.random.seed(10)
    torch.manual_seed(10)
    train_loader = read_data(train_path)
    test_loader = read_data(test_path)
    print("Load Success")

    import lr

    global_model = lr.LROnMedical()
    w_optimizer = MySGD(global_model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)

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
        for inputs, targets in train_loader:
            w_optimizer.zero_grad()
            inputs = inputs.float()
            output = global_model(inputs)
            loss = criterion(output, targets)
            loss.backward()
            w_optimizer.step()
            _, predicted = output.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            average_loss += loss / E

            # Store predicted and true labels for ROC curve
            y_true += targets.tolist()
            y_score += output.tolist()

        print("Training epoch:", epoch, 'acc: %.3f%% (%d/%d)' % (100. * correct / total, correct, total), 'loss: ',
              average_loss)

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

    validation_path = 'data/validation.xlsx'
    validation_loader = read_data(validation_path)

    y_true_val = []
    y_score_val = []

    total_val = 0
    correct_val = 0
    average_loss_val = 0
    global_model.eval()
    for inputs, targets in validation_loader:
        inputs = inputs.float()
        w_optimizer.zero_grad()
        output = global_model(inputs.float())
        loss = criterion(output, targets)

        _, predicted = output.max(1)
        total_val += targets.size(0)
        correct_val += predicted.eq(targets).sum().item()
        average_loss_val += loss.item()

        # Store predicted and true labels for ROC curve
        y_true_val += targets.tolist()
        y_score_val += output.tolist()

    print('Validation acc: %.3f%% (%d/%d)' % (100. * correct_val / total_val, correct_val, total_val), 'loss: ', average_loss_val)

    # Compute ROC curve and AUC for外部验证数据集
    y_true_val = np.array(y_true_val)
    y_score_val = np.array(y_score_val)
    fpr_val, tpr_val, thresholds_val = roc_curve(y_true_val, y_score_val)
    roc_auc_val = auc(fpr_val, tpr_val)

    # Plot ROC curve for external validation
    plt.figure()
    plt.plot(fpr_val, tpr_val, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc_val)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (External Validation)')
    plt.legend(loc="lower right")
    plt.show()

    # Output predicted values and true values for external validation
    predicted_labels_val = np.argmax(y_score_val, axis=1)
    print("Predicted Labels (Validation):", predicted_labels_val)
    print("True Labels (Validation):", y_true_val)
