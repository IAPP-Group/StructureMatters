from argparse import ArgumentParser
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from dataloader import Vid2Graph
from model import Classifier
import torch
import torch.nn.functional as F
import dgl
import numpy as np
from math import inf, ceil
import os
from datetime import datetime


MY_FEATURES_INDEX = {
        "xy": [2],
        "xy-type": [8],
        "xy-split": [2, -5],
        "xy-type-split": [13]
    }

def train(name, data, train_split_name, test_split_name, valid_split_name, features_name):
    
    local_features_name = "-".join([i for i in features_name.split("-") if i=="xy" or i=="type" or i=="split"])
    #print(features_name, local_features_name)
    if features_name != "base":
        features_idxs = MY_FEATURES_INDEX[local_features_name]
        output_weights = f'weights/{features_name}/{name}.pth'
    else:
        output_weights = f'weights/{name}.pth'
    
    wandb.__name__ = name
    
    inner_dim = 35
    lr = 0.005
    num_epochs = 100
    batch_size = 3
    random_state = 46

    print(f"Settings:\n - {inner_dim}\n - {lr}\n - {num_epochs}\n - {batch_size}\n - {random_state}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #print(train_split_name)
    training = Vid2Graph(name=data, split=train_split_name)
    training_dataloader = dgl.dataloading.GraphDataLoader(training, batch_size=batch_size, shuffle=True, drop_last=False)

    #print(len(training))   
 
    testing = Vid2Graph(name=data, split=test_split_name)
    validation = Vid2Graph(name=data, split=valid_split_name)
    
    if features_name != "base":
        node_dim = np.abs(features_idxs).sum()
    else:
        node_dim = training[0][0].ndata['x'].shape[1]
    num_classes = len(training.label_to_int)
    model = Classifier(node_dim, inner_dim, num_classes)
    model.to(device)
    
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss = torch.nn.CrossEntropyLoss()

    wandb.config = {
        "learning_rate": lr,
        "epochs": num_epochs,
        "batch_size": batch_size,
        "inner_dim": inner_dim,
        "random_state": random_state
    }

    best_loss = inf
    best_acc = 0
    batch_per_epoch = ceil(len(training) / batch_size)

    for epoch in range(num_epochs):

        train_loss = 0
        model.train()

        for batch_graphs, labels in training_dataloader:
            if features_name == "base":
                feats = batch_graphs.ndata['x'].type(dtype=torch.float32)
            elif len(features_idxs) == 1:
                feats = batch_graphs.ndata['x'][:,:features_idxs[0]].type(dtype=torch.float32)
            else:
                tmp = torch.cat((batch_graphs.ndata['x'][:,:features_idxs[0]], batch_graphs.ndata['x'][:,features_idxs[1]:]),dim=1)
                assert tmp.shape[1] == node_dim
                feats = tmp.type(dtype=torch.float32)
            #print(feats.shape)
            logits = model(batch_graphs.to(device), feats.to(device))            
            # print(logits.size(), labels.size(), max(labels))
            running_loss = loss(logits, labels.to(device))
            opt.zero_grad()
            running_loss.backward()
            opt.step()

            train_loss += running_loss

        train_loss /= batch_per_epoch
        
        # TO VALIDATE
        model.eval()
        with torch.no_grad():
            graphs = dgl.batch(validation.graphs)
            if features_name == "base":
                feats = graphs.ndata['x'].type(dtype=torch.float32)
            elif len(features_idxs) == 1:
                feats = graphs.ndata['x'][:,:features_idxs[0]].type(dtype=torch.float32)
            else:
                tmp_valid = torch.cat((graphs.ndata['x'][:,:features_idxs[0]], graphs.ndata['x'][:,features_idxs[1]:]),dim=1)
                assert tmp_valid.shape[1] == node_dim
                feats = tmp_valid.type(dtype=torch.float32)
            #print(feats.shape)
            logits = model(graphs.to(device), feats.to(device))
            valid_loss = loss(logits, validation.labels.to(device))
            y_pred = np.argmax(logits.cpu().detach().numpy(), axis=1)
            valid_acc = accuracy_score(validation.labels, y_pred)

            if valid_loss < best_loss and valid_acc >= best_acc:
                best_loss = valid_loss
                best_acc = valid_acc
                torch.save(model.state_dict(), output_weights)
                
                
            wandb.log({
                "train-loss": train_loss,
                "valid-loss": valid_loss,
                "valid-acc": valid_acc
                })

            if epoch % 10 == 0:
                print(f"Epoch {epoch} - Train: Loss {running_loss} - Valid: Loss {valid_loss}, Acc {valid_acc}")
    
    # TO TEST
    # load state dict best weights
    model.load_state_dict(torch.load(output_weights))
    model.eval()
    with torch.no_grad():
        graphs = dgl.batch(testing.graphs)
        if features_name == "base":
            feats = graphs.ndata['x'].type(dtype=torch.float32)
        elif len(features_idxs) == 1:
            feats = graphs.ndata['x'][:,:features_idxs[0]].type(dtype=torch.float32)
        else:
            tmp_test = torch.cat((graphs.ndata['x'][:,:features_idxs[0]], graphs.ndata['x'][:,features_idxs[1]:]),dim=1)
            assert tmp_test.shape[1] == node_dim
            feats = tmp_test.type(dtype=torch.float32)
        #print(feats.shape)
        logits = model(graphs.to(device), feats.to(device))
        test_loss = loss(logits, testing.labels.to(device))
        y_pred = np.argmax(logits.cpu().detach().numpy(), axis=1)
        test_acc = accuracy_score(testing.labels, y_pred)

        print(f"Test: Loss {test_loss}, Acc {test_acc}")
        wandb.log({
            "test_loss":test_loss,
            "test_acc":test_acc
            })    


def get_parser():
    parser = ArgumentParser(
                        prog="train.py",
                        description="GNN analysis on social media identification.")
    parser.add_argument('--dataset_name', type=str, help='Dataset to evaluate Premier-social/Premier-ffmpeg/Premier-avidemux/. ', required=True)
    parser.add_argument('--train_name', type=str, help='Training csv filename w/o extension.', default="train")
    parser.add_argument('--test_name', type=str, help='Test csv filename w/o extension.', default="test")
    parser.add_argument('--valid_name', type=str, help='Validation csv filename w/o extension.', default="valid")
    parser.add_argument('--features', type=str, help='Which features to use, [xy, xy-type, xy-split, xy-type-split]. If not specified all features will be considered.', default="base")
    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

        
    if not os.path.exists(f'weights/{args.features}/'):
        os.makedirs(f'weights/{args.features}/')
    

    experiment = datetime.now()
    dev = args.test_name.split("_")[0]
    experiment = "{}-{}-{}".format(args.dataset_name, dev, experiment.strftime("%m%d-%H%M"))
    valid_pth = [i for i in os.listdir("weights") if i.startswith(f"{args.dataset_name}-{dev}")]    

    if len(valid_pth) == 0 :
        train(experiment, args.dataset_name, args.train_name, args.test_name, args.valid_name, args.features)
    else:
        print("Weights already present!!")
