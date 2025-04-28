# E-UPMiM
Official PyTorch Implementation for Our Paper "Beyond Preferences: Enriching User Profiles for Effective E-commerce Recommendations" (ISCAS 2025)

# Update History
28/04/2025: We updated the GCN following LightGCN, with less parameters and better results.

# Requirements
```
Python = 3.9.16
PyTorch >= 2.0.0
tqdm
numpy
TensorBoard
sklearn
scikit-learn
```
Please note that the TensorFlow implementation of our E-UPMiM has been abandoned as it is hard to maintain. We are sorry for any inconvenience.

# Data Preparations
We have released the processed MovieLens-1M dataset and the checkpoints; for the Fliggy and the Amazon-Toys&Games datasets (and other custom datasets), please download the original data and kindly refer to code/data_preprocess.py to preprocess the data.

# Train
```
python train.py --mode train --dataset movieles --learning_rate 0.001 --topN 10 --model_type E-UPMiM --device cuda:0

python train.py --embedding_dim 64 --hidden_size 64 --num_interest 4 --model_type Comi_Rec --device cuda:0 --learning_rate 0.001 --dataset movieles --mode train
```
Please note that during thesis writing, we have tested more network structures (with better results), and updated the codes accordingly. So it might be different from the original paper. For example, we delete the social networking and time-aware re-ranking as they contribute little to the performances, and replace the original CapsNet with an auto-regressive multi-interest extraction module (less parameters, better results).

For readers' convenience, we also provide a PyTorch implementation of ComiRec, with the same hyper-parameter settings as **[ComiRec](https://github.com/THUDM/ComiRec)**.

# Inference
```
python inference.py
```

# Acknowledgement
The structure of our code is based on **[ComiRec](https://github.com/THUDM/ComiRec)** and **[UMI](https://github.com/WHUIR/UMI)**. The idea of simplifying GCN is from **[LightGCN}(https://github.com/kuandeng/LightGCN)**. We greatly thank their incredible efforts! 

# Citation
If you find our work useful for your research and applications, please kindly consider citing our work:
```
@INPROCEEDINGS{Xu2025User,
  author={Xu, Jingyu and Yang, Zhengwei and and Wang, Zheng},
  booktitle={2025 IEEE International Symposium on Circuits and Systems (ISCAS)}, 
  title={Beyond Preferences: Enriching User Profiles for Effective E-commerce Recommendations}, 
  year={2025},
  volume={},
  number={},
  pages={1-5},
  doi={TBD}}
```
