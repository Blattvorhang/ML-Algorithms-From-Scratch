# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Chinese
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


class DecisionTree:
    """
    Decision Tree classifier using C4.5 algorithm.
    """
    tree: dict              # decision tree using nested dictionary
    numerical_attr: list    # numerical (continuous) attributes
    categorical_attr: list  # categorical (discrete) attributes
    label_name: str         # name of the label
    
    def __init__(self):
        self.tree = None
        self.numerical_attr = None
        self.categorical_attr = None
        self.label_name = None
        
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series):
        self.numerical_attr = X_train.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_attr = X_train.select_dtypes(include=[object]).columns.tolist()
        self.label_name = y_train.name
        self.tree = self.__build_tree(X_train, y_train)
        
    def predict(self, X_test: pd.DataFrame) -> pd.Series:
        return X_test.apply(self.__predict_one, axis=1)
    
    def plot(self):
        plotter = PlotDecisionTree()
        plotter.plot(self.tree)
    
    def __predict_one(self, x: pd.Series):
        node = self.tree
        while isinstance(node, dict):
            key = list(node.keys())[0]
            attr = key.split('≤')[0]
            if attr in self.numerical_attr:
                threshold = float(key.split('≤')[1])
                if x[attr] <= threshold:
                    node = node[key]['是']
                else:
                    node = node[key]['否']
            else:
                node = node[key][x[key]]
        return node
    
    def __build_tree(self, X: pd.DataFrame, y: pd.Series):    
        # if all samples in the same class, return the class
        if y.nunique() == 1:
            return y.iloc[0]
        
        # if no attribute left or all attributes have the same value, 
        # return the majority class
        if X.empty or X.nunique().eq(1).all():
            return y.value_counts().idxmax()
        
        best_attr, best_threshold = self.__choose_best_attr(X, y)
        if best_attr in self.numerical_attr:
            # split the dataset into two parts
            left = X[X[best_attr] <= best_threshold]
            right = X[X[best_attr] > best_threshold]
            # numerical attribute can be used multiple times
            node = {
                f'{best_attr}≤{best_threshold}': {
                    '是': self.__build_tree(left, y.loc[left.index]),
                    '否': self.__build_tree(right, y.loc[right.index])
                }
            }
        else:
            node = {best_attr: {}}
            # split the dataset into multiple parts
            for value in X[best_attr].unique():
                # categorical attribute can be used only once
                sub_X = X[X[best_attr] == value].drop(columns=best_attr)
                sub_y = y.loc[sub_X.index]
                node[best_attr][value] = self.__build_tree(sub_X, sub_y)
        return node
    
    def __choose_best_attr(self, X: pd.DataFrame, y: pd.Series) -> tuple[str, float]:
        best_attr = None
        best_threshold = None
        max_gain_ratio = -np.inf
        for attr in X.columns:
            if attr in self.numerical_attr:
                thresholds = X[attr].unique()
                thresholds.sort()
                for i in range(1, len(thresholds)):
                    threshold = (thresholds[i - 1] + thresholds[i]) / 2
                    gain_ratio = self.__info_gain_ratio(X, y, attr, threshold)
                    if gain_ratio > max_gain_ratio:
                        max_gain_ratio = gain_ratio
                        best_attr = attr
                        best_threshold = threshold
            else:
                gain_ratio = self.__info_gain_ratio(X, y, attr)
                if gain_ratio > max_gain_ratio:
                    max_gain_ratio = gain_ratio
                    best_attr = attr
                    best_threshold = None
        return best_attr, best_threshold

    def __info_gain_ratio(self, X: pd.DataFrame, y: pd.Series, attr, threshold=None):
        info_gain = self.__info_gain(X, y, attr, threshold)
        split_info = self.__split_info(X, attr, threshold)
        # To prevent division by zero
        if split_info == 0:
            return 0
        else:
            return info_gain / split_info

    def __split_info(self, X: pd.DataFrame, attr, threshold=None):
        if threshold is not None:  # for numerical attribute
            left = X[X[attr] <= threshold]
            right = X[X[attr] > threshold]
            left_ratio = len(left) / len(X)
            right_ratio = len(right) / len(X)
            return -left_ratio * np.log2(left_ratio) - right_ratio * np.log2(right_ratio)
        else:  # for categorical attribute
            split_info = 0
            for value in X[attr].unique():
                sub_X = X[X[attr] == value]
                ratio = len(sub_X) / len(X)
                split_info -= ratio * np.log2(ratio)
            return split_info
    
    def __info_gain(self, X: pd.DataFrame, y: pd.Series, attr, threshold=None):
        if threshold is not None:  # for numerical attribute
            left = X[X[attr] <= threshold]
            right = X[X[attr] > threshold]
            left_ratio = len(left) / len(X)
            right_ratio = len(right) / len(X)
            return self.__entropy(y) - (left_ratio * self.__entropy(y.loc[left.index]) +
                                              right_ratio * self.__entropy(y.loc[right.index]))
        else:  # for categorical attribute
            gain = self.__entropy(y)
            for value in X[attr].unique():
                sub_y = y[X[attr] == value]
                ratio = len(sub_y) / len(y)
                gain -= ratio * self.__entropy(sub_y)
            return gain
        
    def __entropy(self, y: pd.Series):
        p = y.value_counts(normalize=True)
        return -np.sum(p * np.log2(p))


class PlotDecisionTree:
    """
    Plot the decision tree.
    It is a modified version from the website: https://zhuanlan.zhihu.com/p/142348406
    """
    decisionNode: dict  # decision node style
    leafNode: dict      # leaf node style
    arrow_args: dict    # arrow style
    ax: plt.Axes        # axes
    totalW: float       # total width of tree
    totalD: float       # total depth of tree
    
    # xOff and yOff are used to track the already drawn nodes 
    # and to place the next node in the right place
    xOff: float         # x offset
    yOff: float         # y offset
    
    def __init__(self):
        # boxstyle: box style, fc: face color
        self.decisionNode = dict(boxstyle="square", fc="0.9")
        self.leafNode = dict(boxstyle="round4", fc="0.8")
        self.arrow_args = dict(arrowstyle="<-")
        
        self.ax = None
        self.totalW = 0
        self.totalD = 0
        self.xOff = 0
        self.yOff = 0
    
    def plot(self, tree: dict):
        fig = plt.figure(1, facecolor='white')
        fig.clf()
        axprps = dict(xticks=[], yticks=[])
        self.ax = plt.subplot(111, frameon=False, **axprps)   # defines the frame and no ticks
        self.totalW = float(self.getNumLeafs(tree))
        self.totalD = float(self.getTreeDepth(tree))   

        self.xOff = -0.5 / self.totalW
        self.yOff = 1.0
        self.__plotTree(tree, (0.5, 1.0), ' ')
        plt.show()
    
    def getNumLeafs(self, tree):
        numLeafs = 0
        firstStr = list(tree.keys())[0]
        secondDict = tree[firstStr]
        for key in secondDict.keys():
            if type(secondDict[key]).__name__ == 'dict':
                numLeafs += self.getNumLeafs(secondDict[key])  # recursive
            else:
                numLeafs += 1
                
        return numLeafs

    def getTreeDepth(self, tree):
        maxDepth = 0
        firstStr = list(tree.keys())[0]
        secondDict = tree[firstStr]
        for key in secondDict.keys():
            if type(secondDict[key]).__name__ == 'dict':
                thisDepth = 1 + self.getTreeDepth(secondDict[key])  # recursive
            else:
                thisDepth = 1
            
            maxDepth = max(maxDepth, thisDepth)
            
        return maxDepth
    
    def __plotNode(self, nodeTxt, centerPt, parentPt, nodeType):
        # The starting point should be moved a little bit to the bottom
        parentPt = (parentPt[0], parentPt[1] - 1.0/self.totalD / 15.0)
        self.ax.annotate(nodeTxt, xy=parentPt, xycoords='axes fraction',  
            xytext=centerPt, textcoords='axes fraction',
            va="center", ha="center", bbox=nodeType, arrowprops=self.arrow_args)
            
    def __plotMidText(self, cntrPt, parentPt, txtString):
        # Calculate the position of the text
        xMid = (parentPt[0]-cntrPt[0])/2.0 + cntrPt[0]
        yMid = (parentPt[1]-cntrPt[1])/2.0 + cntrPt[1]
        self.ax.text(xMid, yMid, txtString, va="center", ha="center")  # rotation=30

    def __plotTree(self, tree, parentPt, nodeTxt):
        # Calculate the width and depth of the current tree, which is
        # different from self.totalW and self.totalD
        numLeafs = self.getNumLeafs(tree)
        depth = self.getTreeDepth(tree)        
        firstStr = list(tree.keys())[0]  # root node
        cntrPt = (self.xOff + (1.0 + float(numLeafs))/2.0/self.totalW, self.yOff)
        
        # Draw the node and the text
        self.__plotMidText(cntrPt, parentPt, nodeTxt)
        self.__plotNode(firstStr, cntrPt, parentPt, self.decisionNode)
        
        secondDict = tree[firstStr]
        self.yOff = self.yOff - 1.0/self.totalD  # reduce y offset
        for key in secondDict.keys():
            if type(secondDict[key]).__name__ == 'dict':
                self.__plotTree(secondDict[key], cntrPt, str(key))  # recursive
            else:
                self.xOff = self.xOff + 1.0/self.totalW  # update x offset
                self.__plotNode(secondDict[key], (self.xOff, self.yOff), cntrPt, self.leafNode)
                self.__plotMidText((self.xOff, self.yOff), cntrPt, str(key))
        self.yOff = self.yOff + 1.0/self.totalD


if __name__ == "__main__":
    columns = ['编号', '色泽', '根蒂', '敲声', '纹理', '脐部', '触感', '密度', '含糖率', '好瓜']
    data = [
        [1,  '青绿', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', 0.697, 0.460, '是'],
        [2,  '乌黑', '蜷缩', '沉闷', '清晰', '凹陷', '硬滑', 0.774, 0.376, '是'],
        [3,  '乌黑', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', 0.634, 0.264, '是'],
        [4,  '青绿', '蜷缩', '沉闷', '清晰', '凹陷', '硬滑', 0.608, 0.318, '是'],
        [5,  '浅白', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', 0.556, 0.215, '是'],
        [6,  '青绿', '稍蜷', '浊响', '清晰', '稍凹', '软粘', 0.403, 0.237, '是'],
        [7,  '乌黑', '稍蜷', '浊响', '稍糊', '稍凹', '软粘', 0.481, 0.149, '是'],
        [8,  '乌黑', '稍蜷', '浊响', '清晰', '稍凹', '硬滑', 0.437, 0.211, '是'],
        
        [9,  '乌黑', '稍蜷', '沉闷', '稍糊', '稍凹', '硬滑', 0.666, 0.091, '否'],
        [10, '青绿', '硬挺', '清脆', '清晰', '平坦', '软粘', 0.243, 0.267, '否'],
        [11, '浅白', '硬挺', '清脆', '模糊', '平坦', '硬滑', 0.245, 0.057, '否'],
        [12, '浅白', '蜷缩', '浊响', '模糊', '平坦', '软粘', 0.343, 0.099, '否'],
        [13, '青绿', '稍蜷', '浊响', '稍糊', '凹陷', '硬滑', 0.639, 0.161, '否'],
        [14, '浅白', '稍蜷', '沉闷', '稍糊', '凹陷', '硬滑', 0.657, 0.198, '否'],
        [15, '乌黑', '稍蜷', '浊响', '清晰', '稍凹', '软粘', 0.360, 0.370, '否'],
        [16, '浅白', '蜷缩', '浊响', '模糊', '平坦', '硬滑', 0.593, 0.042, '否'],
        [17, '青绿', '蜷缩', '沉闷', '稍糊', '稍凹', '硬滑', 0.719, 0.103, '否']
    ]

    label_name = '好瓜'
    df = pd.DataFrame(data, columns=columns)
    df.set_index('编号', inplace=True)
    
    # 去掉“密度属性和编号为9的西瓜”
    df.drop(columns='密度', inplace=True)
    df.drop(index=9, inplace=True)
    
    X = df.drop(columns=label_name)
    y = df[label_name].map({'是': '好瓜', '否': '坏瓜'})
    
    model = DecisionTree()
    model.fit(X, y)
    
    # print(model.tree)
    # print(model.predict(X))
    
    model.plot()
    