import torch
import torchio as tio
from torch.utils.data import DataLoader
import torchvision

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

from lenet import LeNet3D
import resnet
# from IPython import display

from cjltest.utils_model import MySGD

from os import walk

# Image files can be in any format supported by SimpleITK or NiBabel, including DICOM
# subject = tio.datasets.Colin27()

# subject.remove_image('t1')
# print(type(subject.head))
# mask = tio.Mask('brain')

# masked = mask(subject)
# subject.data.plot()
# exit(0)
# subject.add_image(masked.t1, 'Masked')
# subject.plot()

if __name__ == '__main__':
    types = ["ADC", "AQC", "SCC"]
    subjects = []
    subjects_test = []
    train_path = "data/train_processed/Image/"

    np.random.seed(15)
    torch.manual_seed(15)
    train_balance = 120
    for type in types:

        tmp_path = train_path + type
        for (dirpath, dirnames, filenames) in walk(tmp_path):
            # print(filename)
            name_idx = 0
            train_balance = 60
            train_balance = len(filenames)
            while name_idx < train_balance:
                name = filenames[name_idx % len(filenames)]
                name_idx += 1

                if "nrrd" not in name:
                    continue

                print(train_path + type + "/" + name, type + "_" + str(name.split(".")[0]))

                subject_tmp = tio.Subject(
                    scan=tio.ScalarImage(train_path + type + "/" + name),
                    diagnosis=types.index(type),
                    id=type + "_" + str(name.split(".")[0]),
                )

                subjects.append(subject_tmp)

    test_path = "data/test_processed/Image/"

    for type in types:

        tmp_path = test_path + type
        for (dirpath, dirnames, filenames) in walk(tmp_path):
            # print(filename)
            for name in filenames:
                if "nrrd" not in name:
                    continue
                print(test_path + type + "/" + name)
                subject_tmp = tio.Subject(
                    scan=tio.ScalarImage(test_path + type + "/" + name),
                    diagnosis=types.index(type),
                    id=type + "_" + str(name.split(".")[0])
                )

                subjects_test.append(subject_tmp)
    # print(len(subjects))

    training_transform = tio.Compose([
        # tio.ToCanonical(),

        # tio.CropOrPad((40, 30, 40), mask_name='label'),
        # tio.RandomMotion(p=0.02),
        tio.RandomBiasField(p=0.03),
        # tio.Clamp(out_min=-0, out_max=100),
        # tio.ZNormalization(masking_method=lambda x: x > x.mean()),
        # tio.RandomNoise(p=0.5),
        # tio.RandomFlip(),
        # tio.OneOf({
        #     tio.RandomAffine(): 0.8,
        #     tio.RandomElasticDeformation(): 0.2,
        # }),
        # tio.OneHot(),
    ])

    # SubjectsDataset is a subclass of torch.data.utils.Dataset
    subjects_dataset = tio.SubjectsDataset(subjects, transform=training_transform)
    subjects_test_dataset = tio.SubjectsDataset(subjects_test, transform=training_transform)

    # for i in range(0, len(subjects_test_dataset)):
    # print(subjects_dataset[i]['id'], subjects_dataset[i]['scan'][tio.DATA].shape)
    # subjects_test_dataset[i]['scan'].plot()
    # print(subjects_test_dataset[i]['id'], subjects_test_dataset[i]['scan'][tio.DATA].sum())

    #     print(i, torch.sum(subjects_dataset[i].label.data))
    # print(subjects_dataset[0].label.data[0][20][15][20])

    # subjects_dataset[1].plot()
    # subjects_dataset[10].plot()

    # training_batch_size = 8

    k = 24
    training_loader = DataLoader(subjects_dataset, batch_size=4, shuffle=True)
    testing_loader = DataLoader(subjects_test_dataset, batch_size=4, shuffle=True)

    print("finish", len(subjects_dataset), "subjects")

    # global_model = LeNet3D().to("cuda")
    global_model = resnet.generate_model(34, n_input_channels=1, n_classes=3).to("cuda")
    w_optimizer = torch.optim.Adam(global_model.parameters(), lr=0.00001, weight_decay=1e-4)

    criterion = torch.nn.CrossEntropyLoss()
    epoch = 0
    E = 10
    y_true = []
    y_score = []

    for epoch in range(0, 100):
        total = 0
        correct = 0
        batch_idx = 0
        average_loss = 0
        global_model.train()

        for i in range(0, E):
            subjects_batch = next(iter(training_loader))

            # for subjects_batch in training_loader:

            inputs = subjects_batch['scan'][tio.DATA].to("cuda")
            target = subjects_batch['diagnosis'].to("cuda")
            # print(inputs.shape)
            w_optimizer.zero_grad()
            output = global_model(inputs.float())
            # print(output)
            # print(target)
            loss = criterion(output, target)
            loss.backward()
            # for param in global_model.parameters():
            #     print(param.data[0])
            #     break
            # print(loss.grad)

            w_optimizer.step()

            # print("Training epoch:", epoch,'acc: %.3f%% (%d/%d)' % (100. * correct / total, correct, total), 'loss: ', average_loss)

            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            average_loss += loss / E

            # Store predicted and true labels for ROC curve
            y_true += target.tolist()
            y_score += output.tolist()

        # print("Training epoch:", epoch, 'acc: %.3f%% (%d/%d)'
        #         % (100. * correct / total, correct, total), 'loss: ', average_loss)

        total = 0
        correct = 0
        batch_idx = 0
        average_loss = 0
        global_model.eval()
        for subjects_batch in testing_loader:
            # subjects_batch = next(iter(testing_loader))

            # batch_idx += 1
            # print(subjects_batch['id'])
            inputs = subjects_batch['scan'][tio.DATA].to("cuda")
            target = subjects_batch['diagnosis'].to("cuda")
            w_optimizer.zero_grad()
            output = global_model(inputs.float())
            loss = criterion(output, target)

            _, predicted = output.max(1)
            # print(predicted)
            # print(target)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            average_loss += loss / E

            # Store predicted and true labels for ROC curve
            y_true += target.tolist()
            y_score += output.tolist()

        print("Testing epoch:", epoch, 'acc: %.3f%% (%d/%d)'
              % (100. * correct / total, correct, total), 'loss: ', average_loss)

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

validation_scan_path = "data/validation/Image/"
validation_label_path = "data/validation/Label/"
validation_subjects = []

for type in types:
    tmp_path = validation_scan_path + type
    for (dirpath, dirnames, filenames) in walk(tmp_path):
        for name in filenames:
            if "nrrd" not in name:
                continue
            subject_tmp = tio.Subject(
                scan=tio.ScalarImage(validation_scan_path + type + "/" + name),
                label=tio.LabelMap(validation_label_path + type + "/" + name),
                diagnosis=types.index(type),
                id=type + "_" + str(name.split(".")[0]),
                name=name
            )
            transformed = mask(subject_tmp)
            transformed = transform(transformed)
            validation_subjects.append(transformed)

validation_dataset = tio.SubjectsDataset(validation_subjects, transform=training_transform)
validation_loader = DataLoader(validation_dataset, batch_size=4, shuffle=True)

total = 0
correct = 0
average_loss = 0
global_model.eval()
for subjects_batch in validation_loader:
    inputs = subjects_batch['scan'][tio.DATA].to("cuda")
    target = subjects_batch['diagnosis'].to("cuda")
    w_optimizer.zero_grad()
    output = global_model(inputs.float())
    loss = criterion(output, target)

    _, predicted = output.max(1)
    total += target.size(0)
    correct += predicted.eq(target).sum().item()
    average_loss += loss.item()

print('Validation acc: %.3f%% (%d/%d)' % (100. * correct / total, correct, total), 'loss: ', average_loss)


# Compute ROC curve and AUC for external validation
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
