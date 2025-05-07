# SPOT-Trip

This is the official implementation of our paper titled **"Learning Static and Dynamic User Preferences for Out-of-Town Trip Recommendation"**.

Pytorch versions are provided.

> Pytorch: https://pytorch.org

## Data

We have released the travel behavior dataset Foursquare and Yelp which are generated based on the [Foursquare](https://sites.google.com/site/yangdingqi/home/foursquaredataset) and [Yelp](https://www.yelp.com.tw/dataset) dataset. You can run the model with these out-of-town data provided in the respective folder.


## Run Our Model

Simply run the following command to train and evaluate:
```cmd
cd ./code
python main.py --ori_data {...} --dst_data {...} --trans_data {...} --save_path {...} --model SPOT-Trip --mode train --kg --train_trans --ode --s_infer
```
