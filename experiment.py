import ast
import math
import os
import time
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import argparse
from os import listdir
from os.path import isfile, join
from sklearn import svm
from sklearn.metrics import precision_score, recall_score, matthews_corrcoef, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import ComplementNB, GaussianNB
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier

import AF

parser = argparse.ArgumentParser()
parser.add_argument("-X", type=str, help="Path to X data")
parser.add_argument("-y", type=str, help="Path to y data")
parser.add_argument("-timeout", type=int, nargs='?', default=3600000, help="Timeout value in milliseconds")
parser.add_argument("-features", type=str)

args = parser.parse_args()

all_features = ast.literal_eval(args.features)

def evaluate_results(y_test, preds):
    # get TN, FP, FN, TP
    tn, fp, fn, tp = confusion_matrix(y_test, preds, labels=['NO', 'YES']).ravel()

    specificity = tn / (tn + fp)
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp)
    mcc = matthews_corrcoef(y_test, preds)
    recall = tp / (tp + fn)

    print('MCC: %4.3f, Accuracy: %4.3f, Recall (TPR): %4.3f, Specificity (TNR): %4.3f, Precision: %4.3f' % (
        mcc, accuracy, recall, specificity, precision))

    return mcc, accuracy, recall, specificity, precision
    return mcc, accuracy, recall, specificity, precision


def create_solution_vector(semantics, graph_list):
    y = []
    for graph in graph_list:
        solutions = graph.solutions_dict[semantics][0].values()
        for solution in solutions:
            if solution not in ['YES', 'NO']:
                print(f"Wrong value '{solution}' found in graph {graph.file_path}")
            y.append(solution)
    return y


def create_feature_vector(features, graph_list):
    feature_dict = {}

    for feature in features:
        for graph in graph_list:
            fea_temp = graph.feature_dict[feature]
            for node in graph.di_graph:
                key = graph.file_path + "_" + node
                try:
                    item = feature_dict[key]
                    item.extend(list(fea_temp[node]))
                    feature_dict[key] = item
                except KeyError:
                    feature_dict[key] = list(fea_temp[node])
    # Create a DataFrame from the feature_dict and set the column names
    df = pd.DataFrame.from_dict(feature_dict, orient='index')
    df.columns = [f'feature_{i}' for i in range(df.shape[1])]
    # Create solutions vectors
    for problem in problems:
        sol = create_solution_vector(problem, graph_list)
        df[problem] = sol
    return df

#plot class distribution
def plot_dataset(dataset, label_plot, feature):
    num_problems = len(problems)
    factors = []
    for i in range(1, int(math.sqrt(num_problems)) + 1):
        if num_problems % i == 0:
            factors.append((i, num_problems // i))
    cols, rows = factors[-1]

    features_data = [col for col in dataset.columns if 'feature' in col]

    fig, axs = plt.subplots(rows, cols * 2, figsize=(15, 15))

    for i, problem in enumerate(problems):
        # get info on datsets
        sol_data = dataset[problem]
        colors = [mcolors.CSS4_COLORS['purple'] if label == 'YES' else mcolors.CSS4_COLORS['goldenrod'] for label in
                  sol_data]
        # create two new DataFrames for 'YES' and 'NO' labels
        df = dataset[features_data]
        df[problem] = dataset[problem]
        X_yes = df.loc[df[problem] == 'YES'].iloc[:, :-1]
        X_no = df.loc[df[problem] == 'NO'].iloc[:, :-1]
        row = i // cols
        col = i % cols * 2
        ax = axs[row, col]
        if (len(features_data) == 2):
            ax.scatter(X_yes.iloc[:, 0], X_yes.iloc[:, 1], c=mcolors.CSS4_COLORS['purple'], alpha=0.3)
            ax.set_xlabel("in " + feature[0])
            ax.set_ylabel("out " + feature[0])
            ax.set_title(problem)
            ax = axs[row, col + 1]
            ax.scatter(X_no.iloc[:, 0], X_no.iloc[:, 1], c=mcolors.CSS4_COLORS['goldenrod'], alpha=0.3)
            ax.set_xlabel("in " + feature[0])
            ax.set_ylabel("out " + feature[0])
            ax.set_title(problem)
        else:
            ax.hist(X_yes, color=mcolors.CSS4_COLORS['purple'])
            ax.set_ylabel("no arguments")
            ax.set_title(problem + " YES")
            ax = axs[row, col + 1]
            ax.hist(X_no, color=mcolors.CSS4_COLORS['goldenrod'])
            ax.set_ylabel("no arguments")
            ax.set_title(problem + " NO")

    plt.suptitle("Argument Acceptability Distribution for Feature: " + feature[0])
    plt.tight_layout()
    filename_plot = label_plot + " Argument Acceptability Distribution for " + feature[
        0] + " under semantics " + '_'.join(
        problems) + ".jpg"
    # plt.show()
    plt.savefig(filename_plot)


problems = ['DC-PR', 'DC-CO', 'DC-ST', 'DC-GR', 'DS-PR', 'DS-CO', 'DS-ST', 'DS-GR']

# read files for training dataset
mypath = args.X
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f)) and f.endswith(".apx") or f.endswith(".tgf")]
graph_list_train = [AF.AF(mypath + '/' + onlyfiles[i], args.timeout) for i in range(len(onlyfiles))]

yes_nodes = 0
no_nodes = 0
yes_no_nodes = {}
# get overview of data set
print("Training Set")
for problem in problems:
    for g in graph_list_train:
        details = g.solutions_overview()
        yes_nodes += details['semantics'][problem]["yes_no_nodes"][0]
        no_nodes += details['semantics'][problem]["yes_no_nodes"][1]
    print(problem, ": yes:", yes_nodes, "no:", no_nodes)
    # reset yes, no nodes
    yes_nodes = 0
    no_nodes = 0

# graph_details = [g.print_solutions_overview() for g in graph_list_train]
# for g in graph_list_train:
# g.print_solutions()
# print(graph_list_train)

# read file for test dataset
mypath = args.y
onlyfiles_test = [f for f in listdir(mypath) if isfile(join(mypath, f)) and f.endswith(".apx") or f.endswith(".tgf")]
graph_list_test = [AF.AF(mypath + '/' + onlyfiles_test[i], args.timeout) for i in range(len(onlyfiles_test))]
# print(graph_list_test)

# get overview of data set
print("Testing Set")
# get overview of data set
for problem in problems:
    for g in graph_list_test:
        details = g.solutions_overview()
        yes_nodes += details['semantics'][problem]["yes_no_nodes"][0]
        no_nodes += details['semantics'][problem]["yes_no_nodes"][1]
    print(problem, ": yes:", yes_nodes, "no:", no_nodes)
    # reset yes, no nodes
    yes_nodes = 0
    no_nodes = 0

for single_feature in all_features:
    if isinstance(single_feature, list):
        features = single_feature
    else:
        features = [single_feature]

    print(features)
    train_df = create_feature_vector(features, graph_list_train)
    test_df = create_feature_vector(features, graph_list_test)
    train_features = [col for col in train_df.columns if 'feature' in col]
    test_features = [col for col in test_df.columns if 'feature' in col]
    X_train = train_df[train_features]
    X_test = test_df[test_features]

    # plot_dataset(train_df, 'Train', features)
    # plot_dataset(test_df, 'Test', features)

    # normalized
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # run training for all semantics
    test_run_path = 'testruns/'
    test_run_name = test_run_path + 'TRAIN ' + os.path.basename(os.path.normpath(args.X)) + ' TEST ' + os.path.basename(
        os.path.normpath(args.y)) + ' FEATURES ' + '\_'.join(features) + '.txt'
    with open(test_run_name, 'w') as f:
        f.write("\\begin{table}[h]\n")
        f.write("\\centering\n")
        f.write("\\scalebox{0.85}{\n")
        f.write("\\begin{tabular}{|l|l|l|l|l|l|l|l|}\n")
        f.write("\\hline\n")
        f.write("\\multicolumn{8}{|c|}{" + '_'.join(features) + "} \\\\ \\hline\n")
        f.write(
            "\\textbf{Classifier} & \\textbf{MCC} & \\textbf{Accuracy} & \\textbf{TPR} & \\textbf{TNR} & \\textbf{Precision} &  \\textbf{Time Train} &  \\textbf{Time Predict} \\\\ \\hline\n")

        for problem in problems:
            print(problem)
            f.write("\\multicolumn{8}{|c|}{" + problem + "} \\\\ \\hline\n")
            # create solution vector for problem
            y_test = test_df[problem]
            y_train = train_df[problem]

            # training
            print("KNN")
            start = time.time()
            clf = KNeighborsClassifier()
            clf.fit(X_train, y_train)
            end = time.time()
            train_time = end - start
            start = time.time()
            preds = clf.predict(X_test)
            end = time.time()
            test_time = end - start
            mcc, acc, tpr, tnr, prec = evaluate_results(y_test, preds)
            f.write(
                "KNN & {:.3f} & {:.3f} & {:.3f} & {:.3f} & {:.3f} & {:.3f} & {:.3f}\\\\ \\hline\n".format(mcc, acc, tpr,
                                                                                                          tnr, prec,
                                                                                                          train_time,
                                                                                                          test_time))

            print("Naive Bayes")
            start = time.time()
            clf = GaussianNB()
            #clf = ComplementNB(force_alpha=True)
            clf.fit(X_train, y_train)
            end = time.time()
            train_time = end - start
            start = time.time()
            preds = clf.predict(X_test)
            end = time.time()
            test_time = end - start
            mcc, acc, tpr, tnr, prec = evaluate_results(y_test, preds)
            f.write(
                "NB & {:.3f} & {:.3f} & {:.3f} & {:.3f} & {:.3f} & {:.3f} & {:.3f}\\\\ \\hline\n".format(mcc, acc, tpr,
                                                                                                         tnr,
                                                                                                         prec,
                                                                                                         train_time,
                                                                                                         test_time))

            print("Decision Trees")
            start = time.time()
            clf = DecisionTreeClassifier()
            clf.fit(X_train, y_train)
            end = time.time()
            train_time = end - start
            start = time.time()
            preds = clf.predict(X_test)
            end = time.time()
            test_time = end - start
            mcc, acc, tpr, tnr, prec = evaluate_results(y_test, preds)
            f.write(
                "DT & {:.3f} & {:.3f} & {:.3f} & {:.3f} & {:.3f} & {:.3f} & {:.3f}\\\\ \\hline\n".format(mcc, acc, tpr,
                                                                                                         tnr,
                                                                                                         prec,
                                                                                                         train_time,
                                                                                                         test_time))

            print("Random Forest Classifier")
            start = time.time()
            clf = RandomForestClassifier()
            clf.fit(X_train, y_train)
            end = time.time()
            train_time = end - start
            start = time.time()
            preds = clf.predict(X_test)
            end = time.time()
            test_time = end - start
            mcc, acc, tpr, tnr, prec = evaluate_results(y_test, preds)
            f.write(
                "RF & {:.3f} & {:.3f} & {:.3f} & {:.3f} & {:.3f} & {:.3f} & {:.3f}\\\\ \\hline\n".format(mcc, acc, tpr,
                                                                                                         tnr,
                                                                                                         prec,
                                                                                                         train_time,
                                                                                                         test_time))

            print("SVM lin")
            start = time.time()
            clf = svm.LinearSVC()
            clf.fit(X_train, y_train)
            end = time.time()
            train_time = end - start
            start = time.time()
            preds = clf.predict(X_test)
            end = time.time()
            test_time = end - start
            mcc, acc, tpr, tnr, prec = evaluate_results(y_test, preds)
            f.write(
                "SVM lin & {:.3f} & {:.3f} & {:.3f} & {:.3f} & {:.3f} & {:.3f} & {:.3f}\\\\ \\hline\n".format(mcc, acc,
                                                                                                              tpr,
                                                                                                              tnr, prec,
                                                                                                              train_time,
                                                                                                              test_time))

            # print("SVM rbg")
            # start = time.time()
            # clf = SVC()
            # clf.fit(X_train, y_train)
            # end = time.time()
            # train_time = end - start
            # start = time.time()
            # preds = clf.predict(X_test)
            # end = time.time()
            # test_time = end - start
            # mcc, acc, tpr, tnr, prec = evaluate_results(y_test, preds)
            # f.write(
            #     "SVM rbg & {:.3f} & {:.3f} & {:.3f} & {:.3f} & {:.3f} & {:.3f} & {:.3f}\\\\ \\hline\n".format(mcc, acc, tpr,
            #                                                                                                   tnr, prec,
            #                                                                                                  train_time,
            #                                                                                                  test_time))

        f.write("\\end{tabular}\n")
        f.write("}\n")
        f.write("\\caption{Results for test set" + os.path.basename(
            os.path.normpath(args.y)) + ". Classifiers trained with training set " + os.path.basename(
            os.path.normpath(args.X)) + ", features used: " + ', '.join(features) + "}\n")
        f.write("\\end{table}\n")
