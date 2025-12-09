<p align="justify">
# A Multi-head attention-based neural network for cancer-specific identification of $m^7G$ modification sites using hybrid features
This repository contains resources related to the research titled **A Multi-head attention-based neural network for cancer-specific identification of $m^7G$ modification sites using hybrid features** In this study, we propose a computational model based on machine leanring and deep learning approches to identify $m^7G$ sites. $N^7$-methylguanosine ($m^7G$) is one of the most prevalent and biologically important RNA modifications. While several computational models exist for $m^7G$ site prediction in human and non-human species, none have been specifically designed to detect $m^7G$ modification sites in cancers. This study addresses this gap by developing a cancer-specific predictive framework for $m^7G$ site identification. We curated four $m^7G$ benchmark datasets from two cancer cell lines (human osteosarcoma and human hepatocellular carcinoma) from the NCBI Gene Expression Omnibus repository and included an additional benchmark dataset from m7GHub v2.0. We developed a novel multi-head attention-based convolutional neural network model (CNN-MHA-TNC) using two feature encoding methods—trinucleotide composition (TNC) and one-hot encoding (OHE). The model’s performance was compared to support vector machine (SVM), random forest (RF), XGBoost, and a 1D CNN across all five datasets. Robustness and generalizability were evaluated using 5-fold cross-validation and independent testing. Model interpretability was examined with t-SNE and SHAP analyses. The proposed CNN-MHA-TNC model achieved superior performance across all five benchmark datasets, with marked improvements in accuracy, sensitivity, specificity, Matthews correlation coefficient, and area under the receiver operating characteristic curve compared to other machine learning and deep learning models. The t-SNE and SHAP analyses provided biological insight into the feature representations and decision-making process of the model. This study presents one of the first cancer-specific computational frameworks for $m^7G$ site prediction. By combining multi-head attention with convolutional neural networks and complementary feature encoding, our approach enhances predictive power and interpretability.

## Proposed Framework
### Benchmark Dataset
In this study, we derived the Homo sapiens $m^7G$ benchmark datasets from National Center for Biotechnology Information (NCBI) Gene Expression Omnibus (GEO) repository (https://www.ncbi.nlm.nih.gov/geo). These datasets consist of four cell lines data: 
GSM5766798 ($D_1$), GSM5766799 ($D_2$), GSM7717369($D_3$), GSM7717370 ($D_4$). Zhao et al. \cite{zhao2023qki} created the ($D_1$) and ($D_2$) datasets to predict internal $m^7G$-modified transcripts and their interactions with the RNA-binding protein Quaking (QKI), providing cell line–specific benchmarks for modeling these modifications

### 5-fold cross-validation
In this study, we used 5-fold cross-validation. The dataset **S** is split into **K** folds (here **K = 5**), represented as **S₁, S₂, …, S₅**.

### Feature Fusion Space
Feature extraction is a crucial step in designing and extracting information patterns from biological sequences. In this work, we employed two feature encoding methods:
-	**k-mer encoding**
- **One-hot encoding**
The k-mer features were first passed through a dense layer, and then the resulting feature maps were concatenated with the CNN feature maps before further processing

### CNN Model
The **Convolutional Neural Network (CNN) with Multi-head Attention Layers** is specifically designed to capture spatial dependencies in the input data.

<p align="center">
<img src="https://github.com/malikmtahir/m7G/blob/main/m7G_Arch.jpg" width="500" height="800">

<p align="center">
System model of the ensemble learning framework highlighting data, feature fusion, and model spaces

## Feature Visualization
To visualize the features, we use **t-SNE** to reduce the dimensionality of the **CNN features, k-mer features, k-mer features via Dense layers, and Dense layer features** from the training set. This will allow us to see how the model's learned representations cluster and separate different data points in a 2D space, helping to interpret the feature learning process.

<p align="center">
<img src="https://github.com/malikmtahir/m7G/blob/main/m7G.png" width="550" height="500">  <p align="right">


### Comparative Analyses against State-of-the-art Models
This comparative study evaluates the developed ensemble model against the cutting-edge model **im7G-DCT [1] and ** focusing on the m6A sites identification problem.
  
### Statistical Analyses
The statistical analysis is based on the **standard error method** that computes the **mean, standard deviation, critical value** (i.e., **z-score**), and **95% confidence interval** for a sample of **accuracy, sensitivity, specificity, MCC,** and **AUC-ROC**. These analyses aim to validate the quantitative results agains the **CNN** model.

  
# References 
The following references are utlized in the comparative analyses.

[1]. Lei, R., Jia, J. and Qin, L., 2025. im7G-DCT: A two-branch strategy model based on improved DenseNet and transformer for m7G site prediction. Computational Biology and Chemistry, 118, p.108473.

# Contact details
## Muhammad Tahir (m.tahir-ra@uwinnipeg.ca)
### Affiliation:
1. **Department of Applied Computer Science, The University of Winnipeg, 515 Portage Ave, R3B 2E9, MB, Canada**
2. **Department of Computer Science, Abdul Wali Khan University, Mardan, Mardan, 23200, Pakistan**

## Qian Liu (qi.liu@uwinnipeg.ca)
### Affiliation:
1. **Department of Applied Computer Science, The University of Winnipeg, 515 Portage Ave, R3B 2E9, MB, Canada**
</p>
