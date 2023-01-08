# GNN-LSTM-based Fusion Model for Structural Dynamic Responses Prediction

## Introduction
In this study, we developed a novel GNN-LSTM-based fusion model framework. It can predict the nonlinear responses history of acceleration, velocity, and displacement for each floor of any SMRF structure between 4 and 7 stories in height.
![Model](./Figures/fusion_model.png)

---

## Results
| | | | |**Acceleration Dataset**| | | | |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| |**NMSE**| | |**PeakError(%)** | | |**R<sup>2</sup>**| |
| |mean|standard deviation| |mean|standard deviation| |mean|standard deviation|
|**GCN**|0.0016|0.0018| |-3.5926|8.7205| |0.9480|0.0574|
|**GAT**|0.001|0.001| |-3.2544|7.4884| |0.967|0.0346|

| | | | |**Velocity Dataset**| | | | |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| |**NMSE**| | |**PeakError(%)** | | |**R<sup>2</sup>**| |
| |mean|standard deviation| |mean|standard deviation| |mean|standard deviation|
|**GCN**|0.0022|0.0022| |-1.2337|7.4020| |0.9521|0.0504|
|**GAT**|0.0015|0.0022| |-1.9009|5.9421| |0.9692|0.0336|

| | | | |**Displacement Dataset**| | | | |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| |**NMSE**| | |**PeakError(%)** | | |**R<sup>2</sup>**| |
| |mean|standard deviation| |mean|standard deviation| |mean|standard deviation|
|**GCN**|0.0039|0.0038| |-4.4077|7.5982| |0.9253|0.0777|
|**GAT**|0.0018|0.003| |-0.8343|6.2273| |0.9625|0.0658|


---

## Installation
- Linux, CUDA>=11.3
- Python>=3.9.7

We recommend you to use Anaconda to create a conda environment:
```
conda env create -f ./environment.yml
```

---

## Download inference data
```
bash download.sh
```
Following the [checkData.ipynb](./checkData.ipynb) to see the detail of the dataset 

---

## Inference
In this repository, I provide the best GAT model for inference. Follow the steps bellow to predict Acceleration, Velocity, Displacement dataset.
- Acceleration Dataset
```
python inference.py --output_dir ./Inference/Acceleration --response_type Acceleration
```

- Velocity Dataset
```
python inference.py --output_dir ./Inference/Velocity --response_type Velocity
```

- Displacement Dataset
```
python inference.py --output_dir ./Inference/Displacement --response_type Displacement
```

---

## Train
1. create folder:
```
mkdir -p ./Results/GCN_LSTM
mkdir -p ./Results/GAT_LSTM
```
2. Go to [train_GCN_LSTM.arg.py](./train_GCN_LSTM_arg.py) or [train_GAT_LSTM.arg.py](./train_GAT_LSTM_arg.py). Then set the training enviornment and learning target
   - --pack_mode: PPS strategy
   - --compression_rate: SC strategy, feel free to try 10, 20, 40
3. Train the model
```
# use GCN as aggregation function
python train_GCN_LSTM.py

# use GAT as aggregation function
python train_GAT_LSTM.py
``` 
4. Test the model
```
# use GCN as aggregation function
python test_GCN_LSTM.py --output_dir <foler path of target GCN_LSTM model>

# use GAT as aggregation function
python test_GAT_LSTM.py --output_dir <foler path of target GAT_LSTM model>
```