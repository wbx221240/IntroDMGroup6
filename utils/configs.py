from embedding.embedding_models import DWT, ARIMA, KME, TS2Vec
from detector.detectors import LOF, OCSVM, iForest, NeuTraLAD
from others.methods import IDK2, TranAD, USAD, AnoTran

embedding_methods = ["DWT", "ARIMA", "KME", "TS2Vec"]
detectors = ["LOF", "OCSVM", "iForest", "NeuTraLAD"]
end2end = ["IDK2", "TranAD", "USAD"]


embedding_mapping = {"DWT": DWT,
                     "ARIMA":ARIMA,
                     "KME":KME,
                     "TS2Vec":TS2Vec}

detector_mapping = {"LOF":LOF,
                    "OCSVM":OCSVM,
                    "iForest":iForest,
                    "NeuTraLAD": NeuTraLAD}

end2end_mapping = {"IDK2":IDK2,
                   "TranAD":TranAD,
                   "USAD":USAD,
                   "AnoTran":AnoTran}

records = ["embedding", "detector", "precision", "recall", "f-score", "auc"]