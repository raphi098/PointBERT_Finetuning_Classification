# Dental_Segmentation

This repository provides the functionallity to train PointBERT and Pointnet2 Deep Learning architectures on the segmentation task. It is based on the original implementations of both which can be found here: [Pointnet2](https://github.com/yanx27/Pointnet_Pointnet2_pytorch) and [Point-BERT](https://github.com/Julie-tang00/Point-BERT).

The script `segmentation_data_prep.py` provides code that will allow to convert colored pointclouds into a dataset feasable to train pointbert and pointnet2 with. To use it create the folder `Dataset_raw` and place the unzippend folder into it. After that adapt the `if __name__ == "__main__"` in `segmentation_data_prep.py` to the correspondig dataset. The labeled data can be downloaded at: https://cloud.hs-augsburg.de/index.php/s/ZFaAfJf2mYjsLnQ  
The same conda environment as in [dVAE Pretraining](https://gitlab.informatik.hs-augsburg.de/aigs3d/project2_rd/dvae_pretraining) can be used. To prepare you data refer to `segmentation_data_prep.py`.

---
# Start Mlflow UI
```bash 
 mlflow ui --host 127.0.0.1 --port 5000
 ```