from Trainer import Trainer
import torch

if __name__ == '__main__':
    print("torch_version:" + torch.__version__)

    # train and test

    model = Trainer()

    load_model_name = "./model/sanfrancisco/best.pkl"
    load_optimizer_name = None 

    model.train(load_optimizer=load_optimizer_name)

    model.eval(load_model=load_model_name)

