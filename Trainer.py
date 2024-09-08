from model_network import TransformerGCN
import yaml
import torch
import data_utils
from torchmetrics.classification import MulticlassAccuracy


class Trainer(object):
    def __init__(self):
        config = yaml.safe_load(open('config.yaml'))

        self.feature_size = config["feature_size"]
        self.embedding_size = config["embedding_size"]
        self.date2vec_size = config["date2vec_size"]
        self.hidden_size = config["hidden_size"]
        self.num_layers = config["num_layers"]
        self.dropout_rate = config["dropout_rate"]
        self.device = "cuda:" + str(config["cuda"])
        self.learning_rate = config["learning_rate"]
        self.epochs = config["epochs"]

        self.train_batch = config["train_batch"]
        self.vali_batch = config["vali_batch"]
        self.test_batch = config["test_batch"]
        self.traj_file = str(config["traj_file"])

        self.dataset = str(config["dataset"])
        self.early_stop = config["early_stop"]
        
    def eval(self, load_model=None):
        net = TransformerGCN(feature_size=self.feature_size,
                               embedding_size=self.embedding_size,
                               date2vec_size = self.date2vec_size,
                               hidden_size=self.hidden_size,
                               num_layers=self.num_layers,
                               dropout_rate=self.dropout_rate,
                               device=self.device)

        if load_model != None:
            net.load_state_dict(torch.load(load_model))
            net.to(self.device)

            dataload = data_utils.DataLoader(load_part= 'test')
            road_network = data_utils.load_network(self.dataset).to(self.device)

            bt_num = int(dataload.return_triplets_num() / self.test_batch)

            with torch.no_grad():
                acc = 0.0
                for bt in range(bt_num):
                    node_batch = torch.tensor(dataload.getbatch_one(),dtype=torch.int64).to(self.device)
                    nodes = net(road_network, node_batch)
                    
                    metric = MulticlassAccuracy(num_classes=self.hidden_size, ignore_index = -1).to(self.device)
                    acc += metric(nodes.view(-1, nodes.size(-1)), node_batch.view(-1))

                acc /= bt_num
                print("Test Set ACC:" + str(acc.item()*100) + "%")

    def train(self, load_model=None, load_optimizer=None):

        net = TransformerGCN(feature_size=self.feature_size,
                               embedding_size=self.embedding_size,
                               date2vec_size = self.date2vec_size,
                               hidden_size=self.hidden_size,
                               dropout_rate=self.dropout_rate,
                               num_layers=self.num_layers,
                               device=self.device)

        dataload = data_utils.DataLoader(load_part= 'train')
        validload = data_utils.DataLoader(load_part= 'vali')

        optimizer = torch.optim.Adam([p for p in net.parameters() if p.requires_grad], lr=self.learning_rate,
                                     weight_decay=0.0001)
        lossfunction = data_utils.CombinedLoss(dataset=self.dataset)

        net.to(self.device)
        lossfunction.to(self.device)

        road_network = data_utils.load_network(self.dataset).to(self.device)

        bt_num = int(dataload.return_triplets_num() / self.train_batch)

        valid_num = int(validload.return_triplets_num() / self.vali_batch)

        best_epoch = 0
        lastepoch = '0'
        best_hr10 = 0
        if load_model != None:
            net.load_state_dict(torch.load(load_model))
            optimizer.load_state_dict(torch.load(load_optimizer))
            lastepoch = load_model.split('/')[-1].split('_')[3]
            # best_epoch = int(lastepoch)
        for epoch in range(int(lastepoch), self.epochs):
            net.train()
            for bt in range(bt_num):
                node_batch = torch.tensor(dataload.getbatch_one(),dtype=torch.int64).to(self.device)
                nodes = net(road_network, node_batch)
                loss = lossfunction(nodes.view(-1, nodes.size(-1)), node_batch.view(-1))
                # loss.requires_grad = True
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            print('epoch:', epoch,'loss:',loss.item())
            if epoch%2 == 0:
                net.eval()
                with torch.no_grad():
                    acc = 0.0
                    for bt in range(valid_num):
                        node_batch = torch.tensor(validload.getbatch_one(),dtype=torch.int64).to(self.device)
                        nodes = net(road_network, node_batch)
                        
                        metric = MulticlassAccuracy(num_classes=self.hidden_size, ignore_index = -1).to(self.device)
                        # print(node_batch.shape, nodes.shape)
                        acc += metric(nodes.view(-1, nodes.size(-1)), node_batch.view(-1))

                    acc /= valid_num
                    print("\t\tEvaluation ACC: " + str(acc.item()*100) + "%")

                    if acc > best_hr10:
                        best_hr10 = acc
                        best_epoch = epoch
                        # save model
                        save_modelname = './model/{}/best.pkl'.format(self.dataset)
                        torch.save(net.state_dict(), save_modelname)
                    if epoch - best_epoch >= self.early_stop:
                        break
                    '''
                    save_optname = './optimizer/{}/tdrive_TP_2w_ST/{}_{}_epoch_{}.pkl'.format(self.dataset, self.dataset,
                                                                                              self.distance_type,
                                                                                              str(epoch))
                    torch.save(optimizer.state_dict(), save_optname)
                    '''

