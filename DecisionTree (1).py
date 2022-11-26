import matplotlib.pyplot as plt
import numpy as np
import seaborn as seaborn

import csv
import statistics
from sklearn import tree
from sklearn.metrics import accuracy_score, confusion_matrix, plot_confusion_matrix
from sklearn.model_selection import GridSearchCV


def handleDataset(Location):  # 切割分成 data 與 label
    numList_1 = []
    numList_2 = []
    list_data = []
    list_label = []

    csvfile = open(Location)
    lines = csv.reader(csvfile)
    list_0 = list(lines)[1:]

    for n in list_0:  # 將 list 內的字串轉回浮點數
        for m in n:
            numList_1.append(float(m))
        numList_2.append(numList_1)
        numList_1 = []
    list_0 = numList_2

    for n in range(len(list_0)):  # 取得 data
        list_data.append(list_0[n][0:len(list_0[0]) - 1])

    for n in range(len(list_0)):
        list_label.append(list_0[n][len(list_0[0]) - 1])  # 取得 label
    return list_data, list_label


def R_C_swap(data):  # list 行列互換

    data_swap = []
    for n in range(len(data[0])):
        temp = []
        for m in data:
            temp.append(m[n])
        data_swap.append(temp)
    return data_swap


def standardization(list_data):  # 資料標準化
    global sigma_all
    global average_all

    sigma_all = []
    average_all = []
    list_data_swap = R_C_swap(list_data)

    for n in range(len(list_data_swap)):
        sigma_all.append(statistics.pstdev(list_data_swap[n]))  # 8 筆資料的 sigma 陣列
        average_all.append(sum(list_data_swap[n]) / len(list_data_swap[n]))  # 8 筆資料的 average 陣列
        for m in range(len(list_data_swap[n])):
            list_data_swap[n][m] = (list_data_swap[n][m] - average_all[n]) / sigma_all[n]  # 資料標準化

    list_data = R_C_swap(list_data_swap)
    return list_data


def outlier(list_data):  # 蓋帽法處理 outlier
    for n in list_data:
        for m in range(len(n)):
            if abs(n[m]) > 3:
                if n[m] >= 0:
                    n[m] = 3

                else:
                    n[m] = -3

            n[m] = (n[m] * sigma_all[m]) + average_all[m]
    return list_data


def main():
    train_data_location = '/Users/kimmy_yo/PycharmProjects/DesicionTree-demo/train_data (1).csv'
    test_data_location = '/Users/kimmy_yo/PycharmProjects/DesicionTree-demo/test_data (1).csv'

    train_list_data, train_list_label = handleDataset(train_data_location)  # 切割分成 data 與 label
    train_list_data_std = standardization(train_list_data)  # 將 data 標準化
    train_list_data_std_out = outlier(train_list_data_std)  # 蓋帽法處理 outlier

    test_list_data, test_list_label = handleDataset(test_data_location)  # 切割分成 data 與 label
    test_list_data_std = standardization(test_list_data)  # 將 data 標準化
    test_list_data_std_out = outlier(test_list_data_std)  # 蓋帽法處理 outlier

    fn = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction",
          "Age"]
    cn = ["0", "1"]

    # 建立訓練隨機森林模型並做預測

    clf = tree.DecisionTreeClassifier(ccp_alpha=0.005, max_depth=5)  # pruning: max_depth, ccl_alpha
    clf.fit(train_list_data_std_out, train_list_label)
    predict = clf.predict(test_list_data_std_out)

    # 印出預測值
    def print_predictions():
        # predictions
        error = 0
        for i in range(len(test_list_label)):
            if predict[i] != test_list_label[i]:
                error += 1
        print(f"error:{error}")

                # print(f"row{i}, predictions: {predict[i]} || actual:{test_list_label[i]}", end="\n")

        # accuracy
        print(f"trainset_accuracy:{clf.score(train_list_data_std_out, train_list_label).round(2)}")
        print(f"testset_accuracy: {clf.score(test_list_data_std_out, test_list_label)}")

    def print_tree():
        text_representation = tree.export_text(clf, feature_names=fn)
        print(text_representation)

    def visialize_decision_tree():
        fig = plt.figure(figsize=(25, 20))
        _ = tree.plot_tree(clf, feature_names=fn, class_names=cn, filled=True)
        fig.savefig("./decision_Tree.png")

    def pre_pruning():
        # use hyperparameter pruning to find the best pre-pruning parameter
        # but we the result isn't better than setting max_depth to 5
        grid_param = {"criterion": ["gini", "entropy"],
                      "splitter": ["best", "random"],
                      "max_depth": range(2, 50, 1),
                      "min_samples_leaf": range(1, 15, 1),
                      "min_samples_split": range(2, 20, 1)
                      }
        grid_search = GridSearchCV(estimator=clf, param_grid=grid_param, cv=5, n_jobs=-1)
        grid_search.fit(train_list_data_std, train_list_label)

        print(grid_search.best_params_)

    def ConfusionMatrix(clf):

        mat = confusion_matrix(y_true=test_list_label, y_pred=predict)
        plot_confusion_matrix(clf,
                              test_list_data_std_out,
                              test_list_label,
                              display_labels=["True", "False"])

        plt.show()
        print(mat)


    def post_pruning():
        path = clf.cost_complexity_pruning_path(train_list_data_std, train_list_label)
        alphas = path['ccp_alphas']

        accuracy_train, accuracy_test = [], []

        for i in alphas:
            my_tree = tree.DecisionTreeClassifier(ccp_alpha=i)

            my_tree.fit(train_list_data_std, train_list_label)
            train_predict = my_tree.predict(train_list_data_std)
            test_predict = my_tree.predict(test_list_data_std)

            accuracy_train.append(accuracy_score(train_list_label, train_predict))
            accuracy_test.append(accuracy_score(test_list_label, test_predict))

        seaborn.set()
        plt.figure(figsize=(14, 7))
        seaborn.lineplot(y=accuracy_train, x=alphas, label="Train Accuracy")
        seaborn.lineplot(y=accuracy_test, x=alphas, label="Test Accuracy")
        plt.xticks(ticks=np.arange(0.00, 0.04, 0.005))
        plt.show()

    # function for the output

    print_tree()
    print_predictions()
    visialize_decision_tree()
    ConfusionMatrix(clf)
    post_pruning()
    # pre_pruning()  

main()