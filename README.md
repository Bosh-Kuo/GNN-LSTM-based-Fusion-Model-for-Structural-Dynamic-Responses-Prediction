# GNN-LSTM-based Fusion Model for Structural Dynamic Responses Prediction

## Download inference data
```
bash download.sh
```
Following the [checkData.ipynb](./checkData.ipynb) to know the detail of the dataset 

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