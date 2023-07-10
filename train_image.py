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
from sklearn.metrics import roc_curve, auc

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
    scan_path = "data/test/Image/"
    label_path = "data/test/Label/"
    save_path = "data/test_processed/Image/"
    # for t in types:
    np.random.seed(10)
    torch.manual_seed(10)
    for type in types:

        tmp_path = scan_path + type
        for (dirpath, dirnames, filenames) in walk(tmp_path):
            # print(filename)
            for name in filenames:
                if "nrrd" not in name:
                    continue
                print(label_path + type + "/" + name, type + "_" + str(name.split(".")[0]))

                subject_tmp = tio.Subject(
                    scan=tio.ScalarImage(scan_path + type + "/" + name),
                    label=tio.LabelMap(label_path + type + "/" + name),
                    diagnosis=types.index(type),
                    id=type + "_" + str(name.split(".")[0]),
                    name=name
                )

                mask = tio.Mask(masking_method='label', outside_value=0)

                transformed = mask(subject_tmp)

                transform = tio.Compose([
                    # tio.Resample(2),
                    tio.Clamp(out_min=0, out_max=100),
                    tio.CropOrPad((40, 40, 10), mask_name='label')
                ])

                transformed = transform(transformed)
                # subjects.append(subject_tmp)
                subjects.append(transformed)

                # break
        # break

    # label_path_test = "data/test/Label/"
    # for type in types:

    #     tmp_path = label_path_test + type
    #     for (dirpath, dirnames, filenames) in walk(tmp_path):
    #         # print(filename)
    #         for name in filenames:
    #             if "nrrd" not in name:
    #                 continue
    #             print(label_path_test + type + "/" + name)
    #             subject_tmp = tio.Subject(
    #                 scan=tio.ScalarImage(label_path_test + type + "/" + name),
    #                 label=tio.LabelMap(label_path_test + type + "/" + name),
    #                 diagnosis=types.index(type),
    #                 id=type + "_" + str(name.split(".")[0])
    #             )

    #             transform = tio.CropOrPad(
    #                (100, 60, 100),
    #                mask_name='label'
    #             )
    #             transformed = transform(subject_tmp)

    #             subjects_test.append(transformed)

    training_transform = tio.Compose([
        # tio.ToCanonical(),

        tio.CropOrPad((40, 40, 10), mask_name='label'),
        # tio.RandomMotion(p=0.2),
        # tio.RandomBiasField(p=0.3),
        # tio.ZNormalization(masking_method=tio.ZNormalization.mean),
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
    # subjects_test_dataset = tio.SubjectsDataset(subjects_test, transform=training_transform)

    for i in range(0, len(subjects_dataset)):
        # subjects_dataset[i]['scan'].plot()
        print(save_path + types[subjects_dataset[i]['diagnosis']] + "/" + subjects_dataset[i]['name'])
        subjects_dataset[i]['scan'].save(
            save_path + types[subjects_dataset[i]['diagnosis']] + "/" + subjects_dataset[i]['name'])

    #     print(i, torch.sum(subjects_dataset[i].label.data))
    # print(subjects_dataset[0].label.data[0][20][15][20])

    # subjects_dataset[1].plot()
    # subjects_dataset[10].plot()

    training_batch_size = 4

    k = 24
    # training_loader = DataLoader(subjects_dataset, batch_size=3, shuffle=True)
    # testing_loader = DataLoader(subjects_test_dataset, batch_size=4, shuffle=True)

    # print("finish", len(subjects_dataset), "subjects")

    # global_model = LeNet3D()
    # w_optimizer = MySGD(global_model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)

    # criterion = torch.nn.CrossEntropyLoss()
    # epoch = 0
    # E = 10
    y_true = []
    y_score = []

    # for epoch in range(0, 39):
    #     total = 0
    #     correct = 0
    #     batch_idx = 0
    #     average_loss = 0
    #     global_model.train()
    #     for i in range(0, E * 3):

    #         subjects_batch = next(iter(training_loader))
    #         # while True:
    #         #     try:
    #         #         subjects_batch = next(iter(training_loader))
    #         #         break
    #         #     except:
    #         #         pass
    #         # for subjects_batch in training_loader:
    #         batch_idx += 1
    #         # print(subjects_batch['diagnosis'])
    #         # print(subjects_batch['diagnosis'].shape)
    #         # print(subjects_batch['scan'][tio.DATA].shape)

    #         inputs = subjects_batch['label'][tio.DATA]
    #         target = subjects_batch['diagnosis']
    #         # print(inputs.shape)
    #         w_optimizer.zero_grad()
    #         output = global_model(inputs.float())
    #         # print(output)
    #         # print(target)
    #         loss = criterion(output, target)
    #         loss.backward()
    #         # for param in global_model.parameters():
    #         #     print(param.data[0])
    #         #     break
    #         # print(loss.grad)

    #         w_optimizer.step()

```python
# print("Training epoch:", epoch, "batch:", batch_idx, "loss", loss.data.item())
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

y_true_val = []
y_score_val = []

total_val = 0
correct_val = 0
average_loss_val = 0
global_model.eval()
for subjects_batch in validation_loader:
    inputs = subjects_batch['scan'][tio.DATA].to("cuda")
    target = subjects_batch['diagnosis'].to("cuda")
    w_optimizer.zero_grad()
    output = global_model(inputs.float())
    loss = criterion(output, target)

    _, predicted = output.max(1)
    total_val += target.size(0)
    correct_val += predicted.eq(target).sum().item()
    average_loss_val += loss.item()

    # Store predicted and true labels for ROC curve
    y_true_val += target.tolist()
    y_score_val += output.tolist()

print('Validation acc: %.3f%% (%d/%d)' % (100. * correct_val / total_val, correct_val, total_val), 'loss: ', average_loss_val)

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
