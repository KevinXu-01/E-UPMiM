# E-UPMiM
Official TensorFlow Implementation for Our Paper "Beyond Preferences: Enriching User Profiles for Effective E-commerce Recommendations" (ISCAS 2025)

# Requirements
```
TensorFlow == 1.15.0
tqdm
faiss-gpu
numpy
TensorBoard
sklearn
scikit-learn
```
For GPUs higher than RTX 30 series, please kindly refer to [NVIDIA/TensorFlow](https://github.com/NVIDIA/tensorflow).

# Data Preparations
We have released the processed MovieLens-1M dataset; for the Fliggy and the Amazon-Toys&Games datasets (and other custom datasets), please download the original data and kindly refer to code/data_preprocess.py to preprocess the data.

# Train
```
python train.py --mode train --dataset movieles --learning_rate 0.001 --topN 10
```

# Inference
```
python inference.py
```

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
