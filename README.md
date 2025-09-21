# (NeurIPS'25) SPOT-Trip: Dual-Preference Driven Out-of-Town Trip Recommendation

This repository contains the code for our NeurIPS 2025 [paper](https://arxiv.org/abs/2506.01705), where we propose **SPOT-Trip**, a dual-preference driven framework for out-of-town trip recommendation.  
Pytorch versions are provided.

> Pytorch: https://pytorch.org

> If you find our work useful in your research, please consider giving a star ⭐ and citing 📖 our paper:

```bibtex
@inproceedings{liu2025spottrip,
  title={{SPOT-Trip: Dual-Preference Driven Out-of-Town Trip Recommendation}},
  author={Yinghui Liu and Hao Miao and Guojiang Shen and Yan Zhao and Xiangjie Kong and Ivan Lee},
  booktitle={{NeurIPS}},
  year={2025}
}
```

## Abstract
Out-of-town trip recommendation aims to generate a sequence of Points of Interest (POIs) for users traveling from their hometowns to previously unvisited regions based on personalized itineraries, e.g., origin, destination, and trip duration. Modeling the complex user preferences--which often exhibit a two-fold nature of static and dynamic interests--is critical for effective recommendations. However, the sparsity of out-of-town check-in data presents significant challenges in capturing such user preferences. Meanwhile, existing methods often conflate the static and dynamic preferences, resulting in suboptimal performance. In this paper, we for the first time systematically study the problem of out-of-town trip recommendation. A novel framework SPOT-Trip is proposed to explicitly learns the dual static-dynamic user preferences. Specifically, to handle scarce data, we construct a POI attribute knowledge graph to enrich the semantic modeling of users’ hometown and out-of-town check-ins, enabling the static preference modeling through attribute relation-aware aggregation. Then, we employ neural ordinary differential equations (ODEs) to capture the continuous evolution of latent dynamic user preferences and innovatively combine a temporal point process to describe the instantaneous probability of each preference behavior. Further, a static-dynamic fusion module is proposed to merge the learned static and dynamic user preferences. Extensive experiments on real data offer insight into the effectiveness of the proposed solutions, showing that SPOT-Trip achieves performance improvement by up to 17.01%.


## Data

We have released the travel behavior dataset Foursquare and Yelp which are generated based on the [Foursquare](https://sites.google.com/site/yangdingqi/home/foursquaredataset) and [Yelp](https://www.yelp.com.tw/dataset) dataset. You can run the model with these out-of-town data provided in the respective folder.

## Run Our Model

Simply run the following command to train and evaluate:
```cmd
cd ./code
python main.py --ori_data {...} --dst_data {...} --trans_data {...} --save_path {...} --model SPOT-Trip --mode train --kg --train_trans --ode --s_infer
```
## Contact Us
For inquiries or further assistance, contact us at LiuYingHui240@outlook.com.
