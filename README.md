# Additive-Manufacturing-Domain-adaptation-for-Bridging-Dissimilar-Process-Maps

Monitoring Of Laser Powder Bed Fusion Process By Bridging Dissimilar Process Maps Using Deep Learning-based Domain Adaptation on Acoustic Emissions

![Graphical abstract](https://github.com/vigneashpandiyan/Additive-Manufacturing-Domain-adaptation-for-Bridging-Dissimilar-Process-Maps/assets/39007209/fb7d9def-e346-4d26-8ea2-b85c9d80f4c3)

# Journal link
https://doi.org/10.1016/j.addma.2024.103974

![LPBF](https://github.com/vigneashpandiyan/Additive-Manufacturing-Contrastive-Learners/assets/39007209/aa6fa98d-a0c8-4424-8fbf-aae661a5bdbd)

# Overview

Advances in sensorization and identification of information embedded inside sensor signatures during manufacturing processes using Machine Learning (ML) algorithms for better decision-making have become critical enablers in building data-driven monitoring systems. In the Laser Powder Bed Fusion (LPBF) process, data-driven-based process monitoring is gaining popularity since it allows for real-time component quality verification. Real-time qualification of the additively manufactured parts has a significant advantage as the cost of conventional post-manufacturing inspection methods can be reduced. Also, corrective actions or build termination could be done to save machine time and resources. However, despite the successful development in addressing monitoring needs in LPBF processes, less research has been paid to the ML model's robustness in decision-making when dealing with variations in data distribution from the laser-material interaction owing to different process spaces. Inspired by the idea of domain adaptation in ML, in this work, we propose a deep learning-based unsupervised domain adaptation technique to tackle shifts in data distribution owing to different process parameter spaces. 

![RMS Plot](https://github.com/vigneashpandiyan/Additive-Manufacturing-Domain-adaptation-for-Bridging-Dissimilar-Process-Maps/assets/39007209/59923a77-4e3e-4890-9d15-1f39471c9cc9)


# Domain Adaptation

The likelihood of the trained deep learning model performing well on the training data (Source domain) on the test data set drawn from the different data distribution (Target domain) is very low, as shown in Figure (a) and Figure (b). Building a generalizable model trained on a single source dataset and applying it to another target dataset with variations to generate accurate classifications and judgments is the aim of domain adaptation, as shown in Figure (c). Transfer learning and domain adaptation are not the same concepts while being closely linked. For example, the input distribution p(X) is mapped to the labels p(Y|X) in a standard classification setup. In transfer learning, the input distribution remains the same while the labels change, and for domain adaptation, the input distribution changes while the labels stay the same. 

![Picture2](https://github.com/vigneashpandiyan/Additive-Manufacturing-Domain-adaptation-for-Bridging-Dissimilar-Process-Maps/assets/39007209/1db03a92-ac25-40bd-94d2-f33bfd11d553)

Finding the shared latent characteristics between the source and target domains and adapting them to lessen the marginal and conditional mismatch in terms of the feature space between domains is the mechanism of domain adaptation. The domain adaptation paradigm has also been used to diagnose bearing faults and anticipate tool wear in manufacturing processes like milling , in addition to image recognition and segmentation applications . These applications inspired us to use this technique for AM. It is difficult to get a discrete statistical distribution in the sensor information against a built quality since the LPBF process map has a large parameter space. Since the parameter spaces in LPBF are continuous, there is a noticeable shift in the sensor signature as it changes. Domain adaption techniques are essential to handle these shifts. In this study, we propose a method in unsupervised domain adaptation exploiting CNNs to infer class labels from an unlabeled data space using statistical characteristics from a data space that is labelled and verified with ground truth based on metallurgical characterization.

![Picture1](https://github.com/vigneashpandiyan/Additive-Manufacturing-Domain-adaptation-for-Bridging-Dissimilar-Process-Maps/assets/39007209/ccba3fce-e82d-4d70-86d1-3542df835e5f)

# Results
![Picture3](https://github.com/vigneashpandiyan/Additive-Manufacturing-Self-Supervised-Bayesian-Representation-Learning-Acoustic-Emission/assets/39007209/b8586ca2-8bb3-441c-8c57-b7680ed507e6)

# Code
```bash
git clone https://github.com/vigneashpandiyan/Additive-Manufacturing-Domain-adaptation-for-Bridging-Dissimilar-Process-Maps
cd Additive-Manufacturing-Domain-adaptation-for-Bridging-Dissimilar-Process-Maps
python ..codes/ CNN_Main.py
python ..codes/ CNN_Domain Adaptation.py
```

# Citation
```
@article{pandiyan2024Monitoring,
  title={Monitoring Of Laser Powder Bed Fusion Process By Bridging Dissimilar Process Maps Using Deep Learning-based Domain Adaptation on Acoustic Emissions},
  author={Pandiyan, Vigneashwara and Wróbel, Rafał  and Roland Axel, Richter and Leparoux, Marc and Leinenbach, Christian and Shevchik, Sergey },
  journal={Additive Manufacturing},
  year={2024},
  publisher={Elsevier}
}
