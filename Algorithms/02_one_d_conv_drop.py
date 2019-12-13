import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
from utils import *
# from distance_functions import*
np.random.seed(0)

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("\n\nCUDA is available...\n\n")
else:
    device = torch.device('cpu')
    print("\n\nCUDA is NOT available...\n\n")

# READING DATA FROM THE DATASET TO PANDAS DATA FRAME
x=pd.read_excel('DSL-StrongPasswordData.xls')

# REMOVING TWO UNWANTED COLUMN FROM THE DATA SET
del x['sessionIndex']
del x['rep']

# EXTRACTING LIST OF USER NAMES FROM THE DATASET (IN THIS CASE 51)
x1=x.groupby('subject')
user_names=x1.subject.unique()

# CREATING A DICTIONARY CALLED 'all_user_data' WHERE DATA IS STORED AS FOLLOWS
# {KEY : VALUE}
# {USER_i : PANDAS DATAFRAME FOR USER i}
# EXAMPLE: {"user1" : data frame for user 1}
user_all_data = {}


# CREATING A DICTIONARY CALLED 'usernames' WHERE ORIGINAL USER NAMES ARE MAPPED TO NEW USER NAMES
# {NEW USER NAME : ORIGINAL USER NAME}
# EXAMPLE: {"user1" : "s002"}
usernames = {}

for i, user in enumerate(user_names):

	# CREATING NAME OF USER
	name = "user"+str(i+1)

	# STORING USER NAME AND DATA FRAMES TO DICTIONARY
	user_all_data[name] = x1.get_group(user[0])
	usernames[name] = user[0]


# CREATING DATA SET FOR 
x,y = create_data_stucture_neural(user_all_data, usernames)
train_x, test_x, train_y,test_y =  train_test_split(x,y, train_size = 0.7, shuffle=True)

total_train_samples = train_x.shape[0]
total_test_samples = test_x.shape[0]

print("x_train", train_x.shape)
print("y_train", train_y.shape)
print("x_test", test_x.shape)
print("y_test", test_y.shape)


# TRAINING PARAMETERS
max_iters = 250
batch_size = 32
learning_rate = 0.001
decay_rate = 0.1
patience = 30
ch1 = 8
ch2 = 16
ch3 = 32
fc1 = 256
fc2 = 128
decay = True

# RESHAPING THE MATRIX
train_x = np.reshape(train_x, (-1,1,31))
test_x = np.reshape(test_x, (-1,1,31))
print(train_x.shape)

# CREATING DATA LOADERS FOR PUSHING DATA INTO NETWORK
train_x_tn = torch.from_numpy(train_x).type(torch.float32)
train_y_tn = torch.from_numpy(train_y).type(torch.LongTensor)
train_dataloader = DataLoader(TensorDataset(train_x_tn, train_y_tn), batch_size=batch_size, shuffle=True)

test_x_tn = torch.from_numpy(test_x).type(torch.float32)
test_y_tn = torch.from_numpy(test_y).type(torch.LongTensor)
test_dataloader = DataLoader(TensorDataset(test_x_tn, test_y_tn), batch_size=batch_size, shuffle=False)


# DEFINING THE NETWORK ARCHITECTURE
class Net(nn.Module):
	def __init__(self,ch1,ch2,ch3,fc1,fc2):
		super(Net, self).__init__()
		self.block1 = nn.Sequential(nn.Conv1d(1, ch1, kernel_size=3,stride=1, padding = 1),
									nn.BatchNorm1d(ch1),
									nn.ReLU(),
									nn.Conv1d(ch1, ch2, kernel_size=3,stride=1, padding = 1),
									nn.BatchNorm1d(ch2),
									nn.ReLU(),
									nn.Conv1d(ch2, ch3, kernel_size=3,stride=1, padding = 1),
									nn.BatchNorm1d(ch3),
									nn.ReLU(),
									nn.Flatten(),

									nn.Linear(31*ch3,fc1),
									nn.BatchNorm1d(ch1),
									# nn.Dropout(0.0),
									nn.ReLU(),

									nn.Linear(fc1,fc2),
									nn.BatchNorm1d(ch2),
									# nn.Dropout(0.0),
									nn.Linear(fc2,51),
									)

		self.Conv1D = nn.Conv1d(1, 16, kernel_size=3,stride=1, padding = 1)
		# self.Conv2D = nn.Conv2d(1, 16, kernel_size=3,stride=1, padding=1, bias=False)

	def forward(self, x):
		x = self.block1(x)
		# x = self.Conv1D(x)
		# x = self.fc1(x)
		# x = self.Sigmoid(x) 
		# x = self.fc2(x)
		# x = self.Sigmoid(x)
		# x = self.fc3(x)
		return x

# model = Net(inputs, hidden1, hidden2, outputs).to(device)
model = Net(ch1,ch2,ch3,fc1,fc2).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
scheduler = ReduceLROnPlateau(optimizer, 'min', patience=patience, verbose=True, factor=decay_rate)

a = torch.rand((1,1,31)).to(device)
print(a.shape)

b = model(a)
print(b.shape)

train_loss = []
train_acc = []
test_loss = []
test_acc = []

for itr in range(max_iters):
    total_loss = 0
    correct = 0
    i= 0
    for i,data in enumerate(train_dataloader):
        x,y = data
        x,y = x.to(device), y.to(device)
        model.zero_grad()
        # print(x.shape)

        # print(x.shape)
        # print(y.shape)        

        # if i % 100 == 0:
        #   # print("batch progress: {:.2f}" .format(i/len(train_dataloader)*100), end="\r")
        # i +=1

        # get output
        targets = y
        y_pred = model(x)
        loss = nn.functional.cross_entropy(y_pred, targets)

        total_loss += loss.item()
        # print(y_pred)
        predicted = torch.argmax(y_pred, 1)
        correct_pred = torch.sum(predicted == targets).item()

        correct += correct_pred

        # backward
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    loss_norm = total_loss/total_train_samples
    acc = correct/total_train_samples
    train_loss.append(loss_norm)
    train_acc.append(acc)


    # # CHECK THE TEST LOSS AND ACCURACY HERE
    with torch.no_grad():
	    test_correct = 0
	    total_test_loss = 0
	    for data in test_dataloader:
	        # get the inputs
	        x, y = data
	        x, y = x.to(device), y.to(device)

	        
	        # get output
	        targets = y
	        y_pred = model(x)
	        loss = nn.functional.cross_entropy(y_pred, targets)

	        total_test_loss += loss 

	        predicted = torch.argmax(y_pred, 1)
	        correct_pred = torch.sum(predicted == targets).item()
	        test_correct += correct_pred
	        # test_correct += predicted.eq(targets.data).cpu().sum().item()

	    acc_test = test_correct/total_test_samples
	    test_loss_norm = total_test_loss/total_test_samples
	    test_loss.append(test_loss_norm)
	    test_acc.append(acc_test)

	    if decay:
	    	scheduler.step(test_loss_norm)



    if itr % 2 == 0:
        # print("itr: {:02d} \t loss: {:.2f} \t acc : {:.2f}".format(itr, total_loss, acc))
        print("itr: {:02d}   tr_loss: {:.4f}   acc : {:.4f}   tst_loss : {:.4f}   test_acc : {:.4f}"\
            .format(itr, loss_norm, acc, test_loss_norm, acc_test))

plt.figure('accuracy')
plt.plot(range(max_iters), train_acc, color='b', label="Train Acc")
plt.plot(range(max_iters), test_acc, color='r', label="Test Acc")
plt.legend()
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.grid()
plt.savefig("./results/acc.jpg", dpi=300, bbox_inches='tight')
plt.show()

plt.figure('loss')
plt.plot(range(max_iters), train_loss, color='b', label="Train Loss")
plt.plot(range(max_iters), test_loss, color='r', label="Test Loss")
plt.legend()
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.grid()
plt.savefig("./results/loss.jpg", dpi=300, bbox_inches='tight')
plt.show()

print('Train accuracy: {}'.format(train_acc[-1]))

# torch.save(model.state_dict(), "./results/q7_1_3_model_parameter.pkl")

# checkpoint = torch.load('q7_1_2_model_parameter.pkl')
# model.load_state_dict(checkpoint)

# # run on test data
# test_correct = 0
# for data in test_dataloader:
# 	x, y = data
# 	x,y = x.to(device), y.to(device)

# 	# get output
# 	targets = y
# 	y_pred = model(x)
# 	loss = nn.functional.cross_entropy(y_pred, targets)

# 	total_test_loss += loss 

# 	predicted = torch.argmax(y_pred, 1)
# 	correct_pred = torch.sum(predicted == targets).item()
# 	test_correct += correct_pred


# run on test data
test_correct = 0
total_targets = []
total_predicted = []

for data in test_dataloader:
	x, y = data
	x,y = x.to(device), y.to(device)

	# get output
	targets = y
	total_targets.append(targets.cpu().numpy())

	y_pred = model(x)
	loss = nn.functional.cross_entropy(y_pred, targets)

	total_test_loss += loss 

	predicted = torch.argmax(y_pred, 1)
	total_predicted.append(predicted.cpu().numpy())
	correct_pred = torch.sum(predicted == targets).item()
	test_correct += correct_pred

# print(total_targets)
total_targets = np.hstack(total_targets)
total_predicted = np.hstack(total_predicted)

# print(total_targets.shape, total_predicted.shape)

test_acc = test_correct/total_test_samples
print('Test accuracy: {}'.format(test_acc))

class_names = (np.unique(total_targets))

# Plot non-normalized confusion matrix
# ax,cm = plot_confusion_matrix(total_targets, total_predicted, classes=class_names,
#                       title='Confusion matrix, without normalization')
# plt.show()

# Plot normalized confusion matrix
ax,cm = plot_confusion_matrix(total_targets, total_predicted, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')




# cm = generate_confusion_matrix(total_targets, total_predicted, class_names)


# test_acc = test_correct/total_test_samples

# print('Test accuracy: {}'.format(test_acc))
