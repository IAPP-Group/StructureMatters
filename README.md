# StructureMatters
Structure matters: analyzing videos via graph neural networks for social media platform attribution

## Accuracy on different Social Networks
The proposed GNN uses 112 frames per video and motion vectors as node connections.The Native class corresponds to original videos. 

| **Class** | **Base** | **FFmpeg** | **Avidemux** |
|-----------|:--------:|:----------:|:------------:|
| Facebook  |   0.92   |    0.92    |     0.91     |
| Instagram |   0.73   |    0.73    |     0.50     |
| Twitter   |   0.83   |    0.83    |     0.88     |
| Youtube   |   1.00   |    1.00    |     0.83     |
| Native    |   0.97   |    0.88    |     0.97     |
