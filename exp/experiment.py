from utils.configs import *
from sklearn.model_selection import train_test_split
from data_provider.data_loader import data_loader
from utils.util import evaluate, result2df
from detector.detectors import detector
from embedding.embedding_models import embedding_model
import os

class Exp_main:
    def __init__(self, Configs):
        self.configs = Configs

    def _get_data(self, pipeline="embedding"):
        # first, split the data into training and 
        return data_loader(self.configs, pipeline)

    def run(self):
        results = []
        # the embedding-detector pipeline
        train_hist_segments, train_labels, test_hist_segments, test_labels = self._get_data("embedding")
        # print(train_hist_segments.shape, train_labels.shape, test_hist_segments.shape, test_labels.shape)
        # for embed in embedding_methods:
        #     embedding_model = self.get_embedding_model(embed)
        #     if embed ==  "TS2Vec":
        #         embedding_model.train(train_hist_segments)
        #     embedded_train = embedding_model.encode(train_hist_segments)
        #     embedded_test = embedding_model.encode(test_hist_segments)
        #     # train, test split
        #     for detect in detectors:
        #         res = [embed, detect]
        #         detector = self.get_detector(detect)
        #         detector.fit(embedded_train)
        #         anomaly_score = detector.anomaly_score(embedded_test)
        #         res += list(evaluate(test_labels, anomaly_score))
        #         results.append(res)
        # END2END dataloader provider
        recon_train_loader, recon_val_loader, recon_test_loader, recon_test_labels = self._get_data("reconstruction")
        pred_train_loader, pred_val_loader, pred_test_loader, pred_test_labels = self._get_data("prediction")
        ano_train_loader, ano_val_loader, ano_test_loader, ano_test_labels = self._get_data("AnoTran")
        idk_X, idk_label, eval_indices = self._get_data("idk2")
        # the reconstruction pipeline
        usad = self.get_end2end("USAD")(self.configs)
        usad.fit(recon_train_loader, recon_val_loader)
        anomaly_score = usad.anomaly_score(recon_test_loader)
        res = ["reconstruction", "USAD"] + list(evaluate(recon_test_labels, anomaly_score))
        print(res)
        results.append(res)
        # the Anomaly Transformer pipeline
        anotran = self.get_end2end("AnoTran")(self.configs)
        anotran.fit(ano_train_loader, ano_val_loader)
        anomaly_score = anotran.anomaly_score(ano_test_loader)
        res = ["attention", "AnoTran"] + list(evaluate(ano_test_labels, anomaly_score))
        print(res)
        results.append(res)
        # IDK2 pipeline
        idk2 = self.get_end2end("IDK2")(self.configs)
        anomaly_score = -idk2.anomaly_score(idk_X)
        res = ["idk", "idk"] + list(evaluate(idk_label[eval_indices], anomaly_score[eval_indices]))
        print(res)
        results.append(res)
        # result to csv
        save_path = os.path.join(self.configs.result_path, self.configs.dataset, self.configs.datafile.split(".")[0]+".csv")
        result2df(results, save_path)
        print(f"Experiments on {self.configs.dataset + "/" +self.configs.datafile} done!")

    def get_embedding_model(self, name)->embedding_model:
        return embedding_mapping[name]

    def get_detector(self, name)->detector:
        return detector_mapping[name]
    
    def get_end2end(self, name)->detector:
        return end2end_mapping[name]
        


    