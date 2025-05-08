class embedding_model:
    def __init__(self, configs):
        self.configs = configs # all the configuration that will be included 
                               # in the specific embedding model.
        self.model = None

    def train(self, train_loader):
        # train model if necessary. (DWT, KME shouldn't use this api)
        pass

    def encode(self, loader):
        pass

class DWT(embedding_model):
    def __init__(self, configs):
        super().__init__(configs)
        
    def encode(self, loader):
        """DWT.encode

        Args:
            loader (np.ndarray):  N * L (N是序列个数, L是时间序列长度)
        Return: 
            np.ndarray, N * D (N是序列个数, D是每一个序列的编码维度, 就是DWT系数的个数)
        """
        pass
        # TODO: DWT编码方法， 将时间序列片段编码成多维向量。参考GPT

class KME(embedding_model):
    def __init__(self, configs):
        super().__init__(configs)

    def encode(self, configs):
        pass
        # TODO: KME编码方法, 同DWT, 参考GPT

class TS2Vec(embedding_model):
    def __init__(self, configs):
        super().__init__(configs)

    def train(self, train_loader):
        """TS2Vec.train

        Args:
            train_loader (np.ndarray): 输入输出同上
        """
        pass
        # TODO: 调用TS2Vec的fit函数(需要设置参数)来训练模型(ts2vec包)

    def encode(self, loader):
        pass

class ARIMA(embedding_model):
    def __init__(self, configs):
        super().__init__(self, configs)
    
    def encode(self, loader):
        pass
        # TODO: ARIMA编码时间序列, 参考GPT


