## Multimodal Intent Discovery from Livestream Videos

PyTorch code for the Findings of NAACL 2022 paper "Multimodal Intent Discovery from Livestream Videos"

#### Requirements:
This code has been tested on torch==1.9.0 and transformers==4.3.2. Other requirements are [moviepy](https://pypi.org/project/moviepy/) for splicing videos.

#### Data:

We are releasing two datasets in this paper:
* **Behance Intent Discovery Dataset** <br>
This is a dataset containing ~20K sentences with manual annotations for tool and creative intents (see paper) and accompanied by timestamps for the livestream video they have been taken from. <br> The files are available in the ```./data/bid/``` folder. <br> Use ```./scripts/download_videos.py``` to download and splice the videos for the timestamps present in the dataset.<br> We follow the [HERO](https://arxiv.org/abs/2005.00200) paper for extracting video representations; see this repository for extraction code.
* **Behance Livestreams Corpus**: This is the larger unlabelled corpus containing nearly 8K full-length videos and their respective transcripts (download scripts coming soon).

#### Models:

The scripts for training the models presented in the paper are available under ```./model/```. <br>

To train the unimodal RoBERTa model on the Behance Intent Discovery dataset, run
```
bash behance_unimodal.sh <GPU_ID>
```


To train the multimodal late fusion RoBERTa model on the Behance Intent Discovery dataset, run:
```
bash behance_late_fusion.sh <feature_type> <path_to_feature_directory> <GPU_ID>
```


To train the multimodal late fusion RoBERTa model on the Behance Intent Discovery dataset, run:
```
bash behance_late_fusion.sh <feature_type> <path_to_feature_directory> <GPU_ID>
```

Dockerized containers for training HERO + Late Fusion and ClipBERT + Late Fusion models are coming soon.

## Acknowledgement:
The code in this repository has been adapted from [BOND](https://github.com/cliang1453/BOND) and [HERO](https://github.com/linjieli222/HERO) codebases.
