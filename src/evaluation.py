from argparse import ArgumentParser
from sklearn.metrics import accuracy_score
from dataloader import Vid2Graph
from model import Classifier
import torch
import dgl
import numpy as np
from math import inf, ceil
import os
import pickle



MY_FEATURES_INDEX = {
        "xy": [2],
        "xy-type": [8],
        "xy-split": [2, -5],
        "xy-type-split": [13]
    }


def get_all_trained_models(features_name, dataset_name="Premier-social"):
    all_models = {}
    for item in os.listdir(f"weights/{features_name}"):
        if item.endswith(".pth") and item.startswith(dataset_name):
            # Premier-social-M28-0901-2213.pth
            dev = item.split("-")[2]
            all_models[dev] = {"model":item, "train": f"{dev}_social_train"}
    return all_models



def evaluate(features_name, data="Premier-social", classes=["social", "ffmpeg", "avidemux"]):

    all_models = get_all_trained_models(features_name)

    inner_dim = 35
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    local_features_name = "-".join([i for i in features_name.split("-") if i=="xy" or i=="type" or i=="split"])
    #print(features_name, local_features_name)
    
    if features_name != "base":
        features_idxs = MY_FEATURES_INDEX[local_features_name]
    if "random" in features_name:
        classes = ["social"]

    for dev_id, dev_info in all_models.items():

        training = Vid2Graph(name=data, split=dev_info["train"])

        if features_name != "base":
            node_dim = np.abs(features_idxs).sum()
        else:
            node_dim = training[0][0].ndata['x'].shape[1]

        num_classes = len(training.label_to_int)
        model = Classifier(node_dim, inner_dim, num_classes)
        model.to(device)

        model.load_state_dict(torch.load(f'weights/{features_name}/{dev_info["model"]}'))
        model.eval()

        result = {}
        for class_id in classes:
            local_results = {}
            if os.path.exists(f"videos/Premier-{class_id}/{dev_id}_{class_id}_test.txt"):
                testing = Vid2Graph(name=f"Premier-{class_id}", split=f"{dev_id}_{class_id}_test")
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
                    
                    logits = model(graphs.to(device), feats.to(device)).cpu().detach().numpy()
                    y_pred = np.argmax(logits, axis=1)
                    test_acc = accuracy_score(testing.labels, y_pred)

                    print(f"{dev_id}-{class_id} | accuracy: {test_acc}")
                    local_results["logits"] = logits
                    local_results["y_pred"] = y_pred
                    local_results["labels"] = testing.labels.cpu().detach().numpy()
                    local_results["accuracy"] = test_acc
            result[class_id] = local_results

        # write pickle output
        result_output = os.path.join("eval-output", features_name, f"{dev_id}.pkl")
        with open(result_output, 'wb') as handle:
            pickle.dump(result, handle)


def get_parser():
    parser = ArgumentParser(
                        prog="evaluation.py",
                        description="Test GNN weights for social media identification.")
    parser.add_argument('--features', type=str, help='Possible values are: ["base", "xy", "xy-type", "xy-split", "xy-type-split"]', default="base")
    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    data_path = os.path.join('eval-output', args.features)
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    evaluate(args.features)

