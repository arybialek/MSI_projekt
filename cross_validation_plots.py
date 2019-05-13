import pandas as pd
import seaborn as sns

sns.set()
import matplotlib.pyplot as plt
import operator

data = pd.read_csv('file.csv', delimiter=",")

mean_val_loss = []
std_val_loss = []
mean_val_acc = []
std_val_acc = []
x = [i for i in range(75)]

for i in range(75):
    mean_val_loss.append(data[data["epoch"] == i]["val_loss"].mean())
    std_val_loss.append(data[data["epoch"] == i]["val_loss"].std())

    mean_val_acc.append(data[data["epoch"] == i]["val_acc"].mean())
    std_val_acc.append(data[data["epoch"] == i]["val_acc"].std())

val_loss_plot = sns.lineplot(x, mean_val_loss, color='blue')
plt.fill_between(x, list(map(operator.sub, mean_val_loss, std_val_loss)),
                 list(map(operator.add, mean_val_loss, std_val_loss)),
                 color='blue', alpha=0.2)
val_loss_plot.set(xlabel='Epochs', ylabel='Validation loss')
plt.title("Mean & std of validation loss.")
plt.savefig('plot1.png')
plt.show()

val_acc_plot = sns.lineplot(x, mean_val_acc, color='gray')
plt.fill_between(x, list(map(operator.sub, mean_val_acc, std_val_acc)),
                 list(map(operator.add, mean_val_acc, std_val_acc)),
                 color='gray', alpha=0.2)
val_loss_plot.set(xlabel='Epochs', ylabel='Validation loss')
plt.title("Mean & std of validation accuracy.")
plt.savefig('plot2.png')
plt.show()
