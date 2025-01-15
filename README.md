# StructureMatters
Structure matters: analyzing videos via graph neural networks for social media platform attribution

A. Gemelli, D.Shullani, D. Baracchi, S. Marinai, A. Piva


Detecting the origin of a digital video within a social network is a critical task that aids law enforcement and intelligence agencies in identifying the creators of misleading visual content. In this research, we introduce an innovative method for identifying the original social network of a video, even when the video has been altered through actions like group of frames removal and file container reconstruction. The proposed method takes advantage of the video encoding’s temporal uniformity, leveraging motion vectors to characterize the specific features associated to various social media platforms. Each video is represented by a graph where nodes correspond to macroblocks. These macroblocks are interconnected by following the inter-prediction rules outlined in the H.264/AVC codec standard. Such a structure can be then classified using a graph neural network to predict the platform on which the video has been shared. Experimental results demonstrate that this approach outperforms both codec- and content-based approaches, underscoring the effectiveness of a structural approach in attributing the social media platform from which videos originated.


## Accuracy on different Social Networks
The proposed GNN uses 112 frames per video and motion vectors as node connections. The Native class corresponds to original videos. 

| **Class** | **Base** | **FFmpeg** | **Avidemux** |
|-----------|:--------:|:----------:|:------------:|
| Facebook  |   0.92   |    0.92    |     0.91     |
| Instagram |   0.73   |    0.73    |     0.50     |
| Twitter   |   0.83   |    0.83    |     0.88     |
| Youtube   |   1.00   |    1.00    |     0.83     |
| Native    |   0.97   |    0.88    |     0.97     |



## Feature and Graph extraction

0. create enviroment to work with `conda env create -f environment.yaml`
1. extract codec-based information for h264 videos: `use podman`

- [download image](https://drive.google.com/drive/folders/1EVW1nOxWo2cdF4-dYCfcn8d6XxXyVw15?usp=sharing) "container/magicamente-mv-image.tar"

- load image from tar (as super-user)
```
podman load --input container/magicamente-mv-image.tar
```

- extract video features from folder
```
podman run --rm -v test/videos:/home/videos:Z -v test/results:/home/results:Z magicamente_mv
```

- for each evaluated video the .json file is stored within a zip in /results


2. extract a graph representation of one video via `src/vid2graph.py`. The following are valid labels: `Facebook, Instagram, Twitter, Youtube, native`

```bash
python3 src/vid2graph.py --video_zip_path test/results/AgenziaANSATwitter0_binary_output_all_info_up.zip --output_path test/graphs/Twitter/ --video_label Twitter --dataset_path test/results/
```


## Premier dataset
TODO


## Graph training

1. build train/test/valid dataset in a leave-one-device-out fashion
```
src/training_splits.ipynb
```

2. train on `Premier` dataset via `src/train.py`:
```bash
python src/train.py --dataset_name "Premier-social" --train_name "${train_name}" --test_name "${test_name}" --valid_name "${valid_name}"
```

## Evaluation

- accuracy evaluation

```
python3 src/evaluation.py --features xy-type-split
```


- the `eval-output` folder contains the testing results for each device in test. The following is an example of D36 with videos of 10 frames.

```
"eval-output/prediction-experiment-10f/xy-type-split-10f/D36.pkl"

{'social': {'logits': array([[ -2.55591   , -23.755878  ,  -5.537538  ,   0.29886204,
            8.849324  ],
         [ -5.7436295 ,  -9.86431   ,  -3.1096563 ,  -3.5502467 ,
            1.8150616 ]], dtype=float32),
  'y_pred': array([4, 4]),
  'labels': array([4, 4]),
  'accuracy': 1.0},
  
 'ffmpeg': {'logits': array([[ -2.555909 , -23.755878 ,  -5.537539 ,   0.2988611,   8.849324 ]],
        dtype=float32),
  'y_pred': array([4]),
  'labels': array([4]),
  'accuracy': 1.0},
  
 'avidemux': {'logits': array([[ -1.9017043 , -18.747044  ,  -4.6117954 ,  -1.198281  ,
            5.8922095 ],
         [-21.718775  ,  -6.2958994 , -14.621407  ,  -3.3868499 ,
            4.3922167 ],
         [ -3.9792366 , -25.777771  ,  -5.660081  ,  -0.09356739,
           10.22045   ]], dtype=float32),
  'y_pred': array([4, 4, 4]),
  'labels': array([4, 4, 4]),
  'accuracy': 1.0}
}
```



## What to cite

```
@inproceedings{gemelli2024structure,
    title={Structure Matters: Analyzing Videos Via Graph Neural Networks for Social Media Platform Attribution},
    author={Gemelli, Andrea and Shullani, Dasara and Baracchi, Daniele and Marinai, Simone and Piva, Alessandro},
    booktitle={ICASSP 2024-2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
    pages={4735--4739},
    year={2024},
    organization={IEEE},
    doi = {10.1109/ICASSP48485.2024.10447089},
}
```


