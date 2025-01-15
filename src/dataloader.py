import os
import torch
import dgl
import zipfile
import json
from tqdm import tqdm
from dgl.data import DGLDataset, save_graphs, load_graphs
from random import randint

class Vid2Graph(DGLDataset):
    """ Template for customizing graph datasets in DGL.

    Parameters
    ----------
    url : str
        URL to download the raw dataset
    raw_dir : str
        Specifying the directory that will store the
        downloaded data or the directory that
        already stores the input data.
        Default: ~/.dgl/
    save_dir : str
        Directory to save the processed dataset.
        Default: the value of `raw_dir`
    force_reload : bool
        Whether to reload the dataset. Default: False
    verbose : bool
        Whether to print out progress information
    """
    def __init__(self,
                 name='',
                 url=None,
                 raw_dir='videos',
                 save_dir='graphs',
                 force_reload=False,
                 verbose=False,
                 split = 'train'):
        
        #self.mb_split_enc = {'[4, 4]': [0,0,1], '[8, 8]': [0,1,0], '[8, 16]': [0,1,1], '[16, 8]': [1,0,0], '[16, 16]': [1,0,1]}
        #self.mb_type_enc = {'P': [0,0,1], "I": [0,1,0], "Skip": [0,1,1], "SI": [1,0,0], "B": [1,0,1], 'Unknown': [1,1,1]}
        self.mb_split_enc = {'[4, 4]': [0,0,0,0,1], '[8, 8]': [0,0,0,1,0], '[8, 16]': [0,0,1,0,0], '[16, 8]': [0,1,0,0,0], '[16, 16]': [1,0,0,0,0]}
        self.mb_type_enc = {'P': [0,0,0,0,0,1], "I": [0,0,0,0,1,0], "Skip": [0,0,0,1,0,0], "SI": [0,0,1,0,0,0], "B": [0,1,0,0,0,0], 'Unknown': [1,0,0,0,0,0]}
        self.label_to_int = {'Facebook': 0, 'Instagram': 1, 'Twitter': 2, 'Youtube': 3, 'native': 4}
        self.int_to_label = {i:l for l,i in self.label_to_int.items()}
        
        self.split = os.path.join(raw_dir, name, f'{split}.txt')

        super(Vid2Graph, self).__init__(name=name,
                                        url=url,
                                        raw_dir=raw_dir,
                                        save_dir=save_dir,
                                        force_reload=force_reload,
                                        verbose=verbose)        

    def download(self):
        # download raw data to local disk
        # TODO if we want to publish the dataset, add here "wget(...)"
        pass

    def process(self):
        # process raw data to graphs, labels, splitting masks
        self._vid_to_graphs(self.raw_path)
        self.load()
            
    def __getitem__(self, idx):
        # get one example by index
        return self.graphs[idx], self.labels[idx]

    def __len__(self):
        # number of data examples
        return len(self.graphs)

    def save(self):
        # save processed data to directory `self.save_path`
        #! saving graphs individually
        pass

    def load(self):
        # load processed data from directory `self.save_path`
        self.graphs, self.labels = list(), list()
        with open(self.split, 'r') as split:
            for graph in split.readlines():
                graph_path = os.path.join(self.save_path, f'{graph.strip()}.bin')
                if not os.path.isfile(graph_path): continue
                graph, dict = load_graphs(graph_path)
                self.graphs.append(graph[0])
                self.labels.append(dict['label'])
        self.labels = torch.concat(self.labels)

    def has_cache(self):
        # calling load or process methods, depending whether graphs already exists or not
        if os.path.exists(self.save_path):
            print(f"> Loading Graphs: {self.split}")
            self.load()
            return True
        else:
            print("> Pre-Processing Graphs!")
            self.process()
            return False

    def _vid_to_graphs(self, src):

        only_once = True
        rand_mb = randint(0, 10000)

        for fld in os.listdir(src):

            print(f"- {fld}")
            fld_path = os.path.join(src, fld)
            if not os.path.isdir(fld_path): continue

            for vid_id, vid in enumerate(os.listdir(fld_path)):
                
                vid_path = os.path.join(fld_path, vid)
                #? unzip, if necessary
                if vid_path.endswith(".zip"):
                    vid_path = self._unzip(fld_path, vid_path)
                
                #? parsing json information
                vid_data = json.load(open(vid_path))
                vid_frames = vid_data['frames']
                max_x, max_y = vid_data['x'], vid_data['y']

                nodes = list()
                u, v = list(), list()

                for frame_id, frame in enumerate(tqdm(vid_frames, desc=f'Video {vid_id}', disable=not self.verbose)):

                    for mb_id, mb in enumerate(frame['macroblocks']):
                        mb_abs = len(frame['macroblocks'])*frame_id + mb_id

                        mb_pos_qp = [round(mb['x'] / max_x, 4), round(mb['y'] / max_y, 4), 
                                     round(mb['qp_y'] / 51, 4), round(mb['qp_uv'] / 51, 4)]

                        mb_type = mb['mb_type'] if "B" not in mb['mb_type'] else "B"
                        mb_type = self.mb_type_enc[mb_type]
                        mb_split = self.mb_split_enc[str(mb['mb_split'])]
                        
                        feats = mb_pos_qp + mb_type + mb_split
                        nodes.append(feats)

                        if mb.get('neighbours'):
                            
                            for ff, mm_list in mb['neighbours'].items():
                                for mm in mm_list:
                                    u.append(mb_abs)
                                    v.append(len(frame['macroblocks'])*int(ff) + mm)

                        if only_once and self.verbose and mb_abs == rand_mb:
                            only_once = False
                            print(mb)
                            print("MB SPLIT:", "True", mb['mb_split'], "- Transformed:", mb_split)
                            print("MB TYPE:", "True", mb['mb_type'], "- Transformed:", mb_type)
                            print("MB QP:", "True", [mb['qp_y'], mb['qp_uv']], "- Transformed:", mb_pos_qp[2:])
                            print("MB POSITION:", "True", [mb['x'], mb['y']], "- Transformed:", mb_pos_qp[:2])

                g = dgl.graph((u , v), num_nodes=len(nodes))
                g.ndata['x'] = torch.tensor(nodes)
                isolated_nodes = ((g.in_degrees() == 0) & (g.out_degrees() == 0)).nonzero().squeeze(1)
                g.remove_nodes(isolated_nodes)

                graph_path = os.path.join(self.save_path, f'{fld}_{os.path.splitext(vid)[0]}.bin')
                label = torch.tensor([self.label_to_int[fld]])
                save_graphs(str(graph_path), g, {'labels': label})

        return
    
    def _unzip(self, fld_path, vid_path):
            with zipfile.ZipFile(vid_path, 'r') as zip_ref:
                zip_ref.extractall(fld_path)
            os.remove(vid_path)
            return os.path.join(fld_path, zip_ref.namelist()[0])
