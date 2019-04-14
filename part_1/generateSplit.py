import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np

RED_CIRCLE_LABELS = [0,1,2,3,4,5,7,8,9,10,15,16]
INPUT_DIR = 'FullIJCNN2013'

def genPieChart(items, labels=None, explode=None, title="Generic Pie Chart", fname='pieChart'):
    fig1, ax1 = plt.subplots()
    ax1.pie(items, explode=explode, labels=labels, shadow=True, autopct='%1.1f', startangle=90)
    ax1.axis('equal')
    plt.title(title)
    plt.savefig(fname + '.png')

def stratifyOnLabel(gt):
    modeLabel = gt.groupby(['filename']) \
              .agg(lambda x: x['label'].value_counts().index[0])['label']
    gt = gt.set_index('filename')#.join(modeLabel)
    gt = gt.join(modeLabel, rsuffix='_mode', how='left')
    gt = gt[~gt.index.duplicated(keep='first')]
    #print(gt.shape)
    #print(gt['label'].value_counts())
    
    X = gt.index.values
    y = gt['label_mode'].values
    stratifier = StratifiedShuffleSplit(n_splits=1, test_size=.4, random_state=99939)
    for train, test in stratifier.split(X, y):
      X_train, y_train = X[train], y[train]
      X_test_temp, y_test_temp = X[test], y[test]
      val_stratifier = StratifiedShuffleSplit(n_splits=1, test_size=.5, random_state=5)
      for test_, val_ in val_stratifier.split(X_test_temp, y_test_temp):
        X_test, y_test = X_test_temp[test_], y_test_temp[test_]
        X_val, y_val = X_test_temp[val_], y_test_temp[val_]
    
    #print(X_train)
    #print(X_val)
    #print(X_test)
    return X_train, X_val, X_test

if __name__ == '__main__':
    ground_truth = pd.read_csv(INPUT_DIR + '/gt.txt',
                                sep=';',
                                header=None,
                                index_col=False,
                                names=['filename', 'Xmin', 'Ymin', 'Xmax', 'Ymax', 'label'])
    old_len = ground_truth.shape[0]
    ground_truth_rc = ground_truth[ground_truth['label'].isin(RED_CIRCLE_LABELS)]
    new_len = ground_truth_rc.shape[0]
    
    #print(old_len)
    #print(new_len)
    genPieChart([old_len - new_len, new_len], ["Other", "Red Circle Signs"], explode=[0, .1])
    train, val, test = stratifyOnLabel(ground_truth_rc)
    gt_train = ground_truth_rc[ground_truth_rc['filename'].isin(train)]
    gt_val = ground_truth_rc[ground_truth_rc['filename'].isin(val)]
    gt_test = ground_truth_rc[ground_truth_rc['filename'].isin(test)]
    
    trainValueCounts = gt_train['label'].value_counts().sort_index()
    #print(trainValueCounts)
    genPieChart(trainValueCounts, trainValueCounts.index, title="Train Sample Class Distribution", fname="trainPieChart")
    
    valValueCounts = gt_val['label'].value_counts().sort_index()
    #print(valValueCounts)
    genPieChart(valValueCounts, valValueCounts.index, title="Validation Sample Class Distribution", fname="valPieChart")
    
    testValueCounts = gt_test['label'].value_counts().sort_index()
    #print(testValueCounts)
    genPieChart(testValueCounts, testValueCounts.index, title="Test Sample Class Distribution", fname="testPieChart")
    
    
    
    gt_train.to_csv(INPUT_DIR + '/gt_train.txt', index=False)
    gt_val.to_csv(INPUT_DIR + '/gt_val.txt', index=False)
    gt_test.to_csv(INPUT_DIR + '/gt_test.txt', index=False)
