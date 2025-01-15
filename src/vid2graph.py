from argparse import ArgumentParser
import os
import torch
import dgl
import zipfile
import json
import numpy as np
from tqdm import tqdm
from dgl.data import DGLDataset, save_graphs
from random import sample, seed
import sys

seed(42)

mb_split_enc = {'[4, 4]': [0,0,0,0,1], '[8, 8]': [0,0,0,1,0], '[8, 16]': [0,0,1,0,0], '[16, 8]': [0,1,0,0,0], '[16, 16]': [1,0,0,0,0]}
mb_type_enc = {'P': [0,0,0,0,0,1], "I": [0,0,0,0,1,0], "Skip": [0,0,0,1,0,0], "SI": [0,0,1,0,0,0], "B": [0,1,0,0,0,0], 'Unknown': [1,0,0,0,0,0]}

def get_parser():
    parser = ArgumentParser(
                        prog="vid2graph.py",
                        description="Converts JSON H264 features in GDL graph.")
    parser.add_argument('--video_zip_path', type=str, help='Video path in zip format.', required=True)
    parser.add_argument('--dataset_path', type=str, help='Dataset directory for video_zip_path.', required=True)
    parser.add_argument('--video_label', type=str, help='Video label.', required=True)
    parser.add_argument('--output_path', type=str, help='Output directory for graph binary files.', required=True)
    parser.add_argument('--nframes', type=int, help='Number of frames to consider in the graph.')
    parser.add_argument('--feature_type', action="append", help="Admissible values are ['qp', 'xy', 'type', 'split']. By default it uses all features.", default=[])
    parser.add_argument('--arcs_random', action="store_true", help="Default all arcs. Otherwise random arcs removal.")
    return parser

if __name__ == "__main__":

    parser = get_parser()
    args = parser.parse_args()

    label_to_int = {'Facebook': 0, 'Instagram': 1, 'Twitter': 2, 'Youtube': 3, 'native': 4}

    graph_path = os.path.join(args.output_path, args.video_zip_path.replace(".zip",".bin"))

    if os.path.exists(graph_path):
        sys.exit(f"{os.path.basename(graph_path)} graph already present!")


    with zipfile.ZipFile(args.video_zip_path) as myzip:
        video_jsonpath = os.path.basename(args.video_zip_path.replace(".zip", ".json"))
        with myzip.open(video_jsonpath) as myfile:
            # parsing json information
            vid_data = json.load(myfile)
            if args.nframes is not None:
                vid_frames = vid_data["frames"][:min(args.nframes, len(vid_data['frames']))]
            else:
                vid_frames = vid_data["frames"]

            max_x = vid_data["frames"][0]["macroblocks"][-1]['x']
            max_y = vid_data["frames"][0]["macroblocks"][-1]['y']

            nodes = list()
            u, v = list(), list()

            for frame_id, frame in enumerate(tqdm(vid_frames, desc=f'Video {video_jsonpath}', disable=True)):

                for mb_id, mb in enumerate(frame['macroblocks']):

                    mb_abs = len(frame['macroblocks'])*frame_id + mb_id

                    mb_pos_qp = [round(mb['x'] / max_x, 4), round(mb['y'] / max_y, 4), 
                                    round(mb['qp_y'] / 51, 4), round(mb['qp_uv'] / 51, 4)]

                    mb_type = mb['mb_type'] if "B" not in mb['mb_type'] else "B"
                    mb_type = mb_type_enc[mb_type]
                    mb_split = mb_split_enc[str(mb['mb_split'])]
                    
 
                    if len(args.feature_type) == 0:
                        feats = mb_pos_qp + mb_type + mb_split
                        #print(len(feats), feats)
                    else:
                        feats = []
                        #print(args.feature_type)
                        if "xy" in args.feature_type:
                            feats += [round(mb['x'] / max_x, 4), round(mb['y'] / max_y, 4)]
                        if "qp" in args.feature_type:
                            feats += [round(mb['qp_y'] / 51, 4), round(mb['qp_uv'] / 51, 4)]
                        if "type" in args.feature_type:
                            feats += mb_type
                        if "split" in args.feature_type:
                            feats += mb_split
                    
                        #print(len(feats), feats)
                    nodes.append(feats)

                    if mb.get('neighbours'):
                        
                        for ff, mm_list in mb['neighbours'].items():
                            for mm in mm_list:
                                u.append(mb_abs)
                                v.append(len(frame['macroblocks'])*int(ff) + mm)


            g = dgl.graph((u , v), num_nodes=len(nodes))
            g.ndata['x'] = torch.tensor(nodes)

            if args.arcs_random:
                num_nodes = g.num_nodes()
                num_edges = g.num_edges()
                u, v = torch.tensor(sample(range(num_nodes), num_edges)), torch.tensor(sample(range(num_nodes), num_edges))
                g.remove_edges(range(num_edges))
                g.add_edges(u, v)

            isolated_nodes = ((g.in_degrees() == 0) & (g.out_degrees() == 0)).nonzero().squeeze(1)
            g.remove_nodes(isolated_nodes)

            graph_path = os.path.join(args.output_path, video_jsonpath.replace(".json",".bin"))
            label = torch.tensor([label_to_int[args.video_label]])
            save_graphs(str(graph_path), g, {'label': label})
