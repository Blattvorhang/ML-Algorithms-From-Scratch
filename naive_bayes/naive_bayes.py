# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Chinese
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


class NaiveBayes():
    numerical_attr: list     # numerical (continuous) attributes
    categorical_attr: list   # categorical (discrete) attributes
    prior: pd.Series         # prior probability
    likelihood: dict         # likelihood probability
    posterior: pd.DataFrame  # posterior probability
    
    def __init__(self) -> None:
        self.numerical_attr = None
        self.categorical_attr = None
        
        self.prior = None
        self.likelihood = None
        self.posterior = None
        
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        self.numerical_attr = X_train.select_dtypes(include=['float64']).columns
        self.categorical_attr = X_train.select_dtypes(include=['object']).columns
        
        # prior probability (Laplacian smoothing)
        num_classes = y_train.nunique()
        self.prior = (y_train.value_counts() + 1) / (len(y_train) + num_classes)
        self.prior = self.prior.reindex(y_train.unique(), fill_value=1 / (len(y_train) + num_classes))
        
        # likelihood probability
        self.likelihood = {}
        for label in self.prior.index:
            self.likelihood[label] = {}
            for attr in self.numerical_attr:
                self.likelihood[label][attr] = {
                    'mean': X_train.loc[y_train == label, attr].mean(),
                    'std': X_train.loc[y_train == label, attr].std()
                }
            for attr in self.categorical_attr:  # Laplacian smoothing
                num_categories = X_train[attr].nunique()
                # reindex ensures that all categories are included
                counts = X_train.loc[y_train == label, attr].value_counts().reindex(X_train[attr].unique(), fill_value=0)
                self.likelihood[label][attr] = (counts + 1) / (len(y_train[y_train == label]) + num_categories)

        # posterior probability
        self.posterior = pd.DataFrame(index=y_train.index, columns=self.prior.index)
        for label in self.prior.index:
            # use log probability to avoid underflow
            self.posterior[label] = np.log(self.prior[label])
            for attr in self.numerical_attr:
                self.posterior[label] += np.log(self.gaussian_pdf(X_train[attr], label, attr))
            for attr in self.categorical_attr:
                self.posterior[label] += np.log(self.categorical_pdf(X_train[attr], label, attr))
        
        # normalization for visualization
        # self.posterior = self.posterior.div(self.posterior.sum(axis=1), axis=0)
        log_sum = np.log(np.exp(self.posterior).sum(axis=1))
        self.posterior = np.exp(self.posterior.sub(log_sum, axis=0))
        
    def gaussian_pdf(self, x: pd.Series, label: str, attr: str) -> pd.Series:
        mean = self.likelihood[label][attr]['mean']
        std = self.likelihood[label][attr]['std']
        return (1 / (np.sqrt(2 * np.pi) * std)) * np.exp(-0.5 * ((x - mean) / std) ** 2)
    
    def categorical_pdf(self, x: pd.Series, label: str, attr: str) -> pd.Series:
        return x.map(self.likelihood[label][attr])
    
    def predict(self, X_test: pd.DataFrame) -> pd.Series:
        posterior = pd.DataFrame(index=X_test.index, columns=self.prior.index)
        for label in self.prior.index:
            # use log probability to avoid underflow
            posterior[label] = np.log(self.prior[label])
            for attr in self.numerical_attr:
                posterior[label] += np.log(self.gaussian_pdf(X_test[attr], label, attr))
            for attr in self.categorical_attr:
                posterior[label] += np.log(self.categorical_pdf(X_test[attr], label, attr))
                
        # print(posterior)
        return posterior.idxmax(axis=1)


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
    
    X_train = df.drop(columns=label_name)
    y_train = df[label_name].map({'是': '好瓜', '否': '坏瓜'})
    
    model = NaiveBayes()
    model.fit(X_train, y_train)
    
    # accuracy
    y_pred = model.predict(X_train)
    accuracy = (y_pred == y_train).mean()
    print(f'Accuracy: {accuracy:.2f}')
    
    # print('Likelihood probability:')
    # print(model.likelihood)
    # print()
    
    # test    
    X_test = pd.DataFrame(
        [['乌黑', '稍蜷', '浊响', '清晰', '稍凹', '软粘', 0.361, 0.371]],
        columns=columns[1:-1])
    y_pred = model.predict(X_test)
    print(y_pred.values[0])
    
    # plot posterior probability
    fig, ax = plt.subplots(figsize=(8, 6))
    model.posterior.plot(kind='bar', ax=ax)
    ax.set_title('各样本后验概率')
    ax.set_xlabel('样本编号')
    ax.set_ylabel('概率')
    plt.show()
    