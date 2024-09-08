# ST2Vec
## Requirements

- Ubuntu OS
- Python >= 3.5 (Anaconda3 is recommended)
- PyTorch 1.4+
- A Nvidia GPU with cuda 10.2+

## Datasets

* Trajectory dataset (TDrive) and Rome are an open source data set
* We provided the road network data and map-matching result data

## Record

- Location set ranking

## Reproducibility & Training

1. Data preprocessing (Time embedding and node embedding)

   ```shell
   python3 preprocess.py
   ```

2. Training

   ```shell
   nohup python3 -u main.py > log.txt 2>&1 &
   ```

3. Visualization

   ```
   python3 -m visualize.visualize
   python3 -m visualize.eval
   ```
