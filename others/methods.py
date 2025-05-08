import numpy as np
from detector.detectors import detector
from .idk.IDK_T import IDK_T
from .usad.usad import UsadModel, training, testing
from .anotran.AnomalyTransformer import AnomalyTransformer
import torch
import torch.nn as nn
from .anotran.utils import *
import time
from .usad.utils import to_device, get_default_device


class IDK2(detector):
    def __init__(self, configs):
        super().__init__(configs)

    def anomaly_score(self, X):
        return IDK_T(X, width=self.configs.window_size, psi1=self.configs.psi1, psi2=self.configs.psi2, t=self.configs.t)


class USAD(detector):
    def __init__(self, configs):
        super().__init__(configs)
        self.device = get_default_device()
        # print((configs.window_size, configs.hist_size))
        self.model = to_device(UsadModel(configs.hist_size, configs.z_dim), self.device)

    def fit(self, train_loader, val_loader):
        self.model = training(self.configs.n_epochs, self.model, train_loader, val_loader)

    def anomaly_score(self, test_loader):
        anomaly_score = testing(self.model, test_loader)
        return anomaly_score

class TranAD(detector):
    def __init__(self, configs):
        super().__init__(configs)

    def fit(self, X):
        pass

    def anomaly_score(self, train_loader, val_loader):
        pass

class AnoTran(detector):
    def __init__(self, configs):
        super().__init__(configs)
        self.model = AnomalyTransformer(win_size=configs.hist_size, enc_in=configs.input_c, c_out=configs.output_c, e_layers=3)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=configs.learning_rate)
        if torch.cuda.is_available():
            self.model.cuda()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.criterion = nn.MSELoss()
        self.k = configs.k

    def fit(self, train_loader, val_loader):
        time_now = time.time()
        path = os.path.join(self.configs.checkpoint, "AnoTran/", self.configs.dataset)
        if not os.path.exists(path):
            os.makedirs(path)
        early_stopping = EarlyStopping(patience=3, verbose=True, datafile_name=self.configs.datafile)
        train_steps = len(train_loader)

        for epoch in range(self.configs.n_epochs):
            iter_count = 0
            loss1_list = []

            epoch_time = time.time()
            self.model.train()
            # print(len(train_loader))
            for i, input_data in enumerate(train_loader):
                
                self.optimizer.zero_grad()
                iter_count += 1
                input = input_data.float().to(self.device)

                output, series, prior, _ = self.model(input)

                # calculate Association discrepancy
                series_loss = 0.0
                prior_loss = 0.0
                for u in range(len(prior)):
                    series_loss += (torch.mean(my_kl_loss(series[u], (
                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                   self.configs.hist_size)).detach())) + torch.mean(
                        my_kl_loss((prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                           self.configs.hist_size)).detach(),
                                   series[u])))
                    prior_loss += (torch.mean(my_kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.configs.hist_size)),
                        series[u].detach())) + torch.mean(
                        my_kl_loss(series[u].detach(), (
                                prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                       self.configs.hist_size)))))
                series_loss = series_loss / len(prior)
                prior_loss = prior_loss / len(prior)

                rec_loss = self.criterion(output, input)

                loss1_list.append((rec_loss - self.k * series_loss).item())
                loss1 = rec_loss - self.k * series_loss
                loss2 = rec_loss + self.k * prior_loss

                if (i + 1) % 100 == 0:
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.configs.n_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                # Minimax strategy
                loss1.backward(retain_graph=True)
                loss2.backward()
                self.optimizer.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(loss1_list)

            vali_loss1, vali_loss2 = self.vali(val_loader)

            print(
                "Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} ".format(
                    epoch + 1, train_steps, train_loss, vali_loss1))
            early_stopping(vali_loss1, vali_loss2, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            adjust_learning_rate(self.optimizer, epoch + 1, self.configs.learning_rate)

    def anomaly_score(self, test_loader):
        self.model.eval()
        temperature = 50
        criterion = nn.MSELoss(reduce=False)

        
        attens_energy = []
        for i, input_data in enumerate(test_loader):
            input = input_data.float().to(self.device)
            output, series, prior, _ = self.model(input)

            loss = torch.mean(criterion(input, output), dim=-1)

            series_loss = 0.0
            prior_loss = 0.0
            for u in range(len(prior)):
                if u == 0:
                    series_loss = my_kl_loss(series[u], (
                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                   self.configs.hist_size)).detach()) * temperature
                    prior_loss = my_kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.configs.hist_size)),
                        series[u].detach()) * temperature
                else:
                    series_loss += my_kl_loss(series[u], (
                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                   self.configs.hist_size)).detach()) * temperature
                    prior_loss += my_kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.configs.hist_size)),
                        series[u].detach()) * temperature
            # Metric
            metric = torch.softmax((-series_loss - prior_loss), dim=-1)
            cri = metric * loss
            cri = cri.detach().cpu().numpy()
            attens_energy.append(cri)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1, self.configs.hist_size)
        test_energy = np.sum(attens_energy, axis=1)
        
        return test_energy
        
        
    def vali(self, val_loader):
        self.model.eval()

        loss_1 = []
        loss_2 = []
        for i, input_data in enumerate(val_loader):
            input = input_data.float().to(self.device)
            output, series, prior, _ = self.model(input)
            series_loss = 0.0
            prior_loss = 0.0
            for u in range(len(prior)):
                series_loss += (torch.mean(my_kl_loss(series[u], (
                        prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                               self.configs.hist_size)).detach())) + torch.mean(
                    my_kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.configs.hist_size)).detach(),
                        series[u])))
                prior_loss += (torch.mean(
                    my_kl_loss((prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                       self.configs.hist_size)),
                               series[u].detach())) + torch.mean(
                    my_kl_loss(series[u].detach(),
                               (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                       self.configs.hist_size)))))
            series_loss = series_loss / len(prior)
            prior_loss = prior_loss / len(prior)

            rec_loss = self.criterion(output, input)
            loss_1.append((rec_loss - self.k * series_loss).item())
            loss_2.append((rec_loss + self.k * prior_loss).item())

        return np.average(loss_1), np.average(loss_2)
