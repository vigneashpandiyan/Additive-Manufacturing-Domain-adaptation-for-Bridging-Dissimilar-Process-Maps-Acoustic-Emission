# Additive-Manufacturing-Domain-adaptation-for-Bridging-Dissimilar-Process-Maps
Monitoring Of Laser Powder Bed Fusion Process By Bridging Dissimilar Process Maps Using Deep Learning-based Domain Adaptation on Acoustic Emissions

# Journal link
https://doi.org/10.1016/j.jmapro.2022.07.033

![LPBF](https://github.com/vigneashpandiyan/Additive-Manufacturing-Contrastive-Learners/assets/39007209/aa6fa98d-a0c8-4424-8fbf-aae661a5bdbd)

# Overview

Although most LPBF monitoring strategies based on process zone information in the literature for LPBF process are trained in supervised, unsupervised and semi-supervised manners, the authors of this work take a first step towards creating a framework for monitoring part quality in terms of build density using a self-supervisedly trained Bayesian Neural Network (BNN). The motivation for this approach stems from the challenges of labeling datasets with discrete process dynamics and semantic complexities. Self-supervised models offer a fully unsupervised training opportunity, which reduces the time and cost associated with algorithm setup. Furthermore, self-supervised learning can facilitate transfer learning, where a pre-trained model is fine-tuned for a specific task, thus minimizing the amount of labeled data required and enhancing efficiency, which is also demonstrated in this work. 



# Domain Adaptation

Deep learning models are created to understand the relationships between data samples in order to make predictions about the objectives for which they were trained. Thanks to recent improvements in self-supervised representation learning, models can now be trained on less annotated data samples. The goal of self-supervised learning is to identify the most informative characteristics of unlabelled data by creating a supervisory signal, which leads to the learning of generalizable representations. Self-supervised learning has been successful in various computer vision tasks. The self-supervised representation introduced in this study draws inspiration from prior works [64-67] and offers a powerful method for decoding inter and intra-temporal relationships. The methodology proposed aims to extract time series representations from unlabeled data through inter-sample and intra-temporal relation reasoning. This is accomplished by utilizing a shared representation learning encoder backbone (f_( θ)) based on Bayesian Neural Network (BNN), as depicted in Figure below. 




# Code
```bash
git clone https://github.com/vigneashpandiyan/Additive-Manufacturing-Domain-adaptation-for-Bridging-Dissimilar-Process-Maps
cd Additive-Manufacturing-Domain-adaptation-for-Bridging-Dissimilar-Process-Maps
python Main.py
```

# Citation
```
@article{pandiyan2022situ,
  title={},
  author={},
  journal={},
  volume={},
  pages={},
  year={},
  publisher={Elsevier}
}
