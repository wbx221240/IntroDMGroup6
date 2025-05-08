class detector:
    def __init__(self, configs):
        self.configs = configs # all the configuration in detector setting.
        self.model = None

    def fit(self, X):
        pass

    def anomaly_score(self, X):
        pass

class LOF(detector):
    def __init__(self, configs):
        super().__init__(configs)
    
    def fit(self, X):
        """LOF.fit(X)

        Args:
            X (np.ndarray): 编码后的时间序列(N * D)
        """
        pass
        # TODO: 基于LOF(sklearn包)训练detector并保存模型

    def anomaly_score(self, X=None):
        """异常分数

        Args:
            X (np.ndarray): 
        """
        pass
        # TODO: LOF.negative_outlier_factor_ # 注意越大越异常还是越小越异常.(所有异常分数都统一为正数, 越大越异常)

class OCSVM(detector):
    def __init__(self, configs):
        super().__init__(configs)
    
    def fit(self, X):
        """同LOF

        Args:
            X (np.ndarray): 编码后的训练集
        """
        pass
        # TODO: 基于OCSVM(sklearn包)训练detector并保存模型

    def anomaly_score(self, X):
        pass
        # TODO: 用训练好的OCSVM模型在测试集上计算anomaly score(ocsvm.score_samples())


class iForest(detector):
    def __init__(self, configs):
        super().__init__(configs)

    def fit(self, X): 
        pass
        # TODO: 同上面其他detector

    def anomaly_score(self, X):
        pass
        # TODO: 同上面其他detector, -iforest.decision_function() from sklearn.ensemble

class NeuTraLAD(detector):
    def __init__(self, configs):
        super().__init__(configs)

    def fit(self, X): 
        pass
        # TODO: copy NeuTralLAD的github仓库. https://github.com/boschresearch/NeuTraL-AD, 具体操作办法见仓库.

    def anomaly_score(self, X):
        pass
        # TODO: 同上面其他detector