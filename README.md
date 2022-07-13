## On Curriculum Learning for Commonsense Reasoning

PyTorch code for the Findings of NAACL 2022 paper "Multimodal Intent Discovery from Livestream Videos"

#### Requirements:
This code has been tested on torch==1.9.0 and transformers==4.3.2. Other requirements are [https://pypi.org/project/moviepy/](moviepy).

#### Data:

We are releasing two datasets in this paper:
* Behance Intent Discovery Dataset: This is a dataset containing ~20K sentences with manual annotations for tool and creative intents (see paper) and accompanied by timestamps for the livestream video they have taken from. The files are available in the ```./data/bid/``` folder. Use ```./scripts/download_videos.py``` to download and splice the videos for the timestamps present in the dataset.
* Behance Livestreams Corpus: This is the larger unlabelled corpus containing nearly 8K full-length videos and their respective transcripts (download scripts coming soon).

#### Models:

The scripts for training the models presented in the paper are available under ```./models/```

