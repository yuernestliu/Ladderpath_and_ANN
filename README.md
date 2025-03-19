# Introduction
This repository includes all the source codes and data for the paper "*Correlating Measures of Hierarchical Structures in Artificial Neural Networks with Their Performance*, npj Complexity, 1, 15, 2024." by Zhuoying Xu et al. https://www.nature.com/articles/s44260-024-00015-x 


## Files
- **main.py**: An entry file imports existing functions and trains neural networks based on various tasks. Subsequently, it generates sequences and calculates the $\eta$ values of these neural networks.

- **ladderpath.py**: Calculates the ladderpath and $\eta$ of a set of sequences (downloaded from https://github.com/yuernestliu/LadderpathCalculator).

- **layer1_function_mod.py**: It transforms the neural network into a set of sequences (for the network with one hidden layer), which enables the calculation of the $\eta$ value. The same for layer2_function_mod.py (two hidden layers), and so on.

- **mod2dim3.xlsx**: The dataset for the odd-even recognition task.

- **Data_SI.xlsx**: Contains all the data for the five different tasks described in the Supplementary Information Section 2.




## Instructions

1. **Required Packages**: 

   Ensure Python 3.7 and Tensorflow 1.14.0 are installed on your system (and also install these packages scipy, keras, networkx, scikit-learn, numpy).

2. **Configure Parameters**:

   Open the `main.py` file and configure parameters such as the dataset path, network structure, and computation epochs as per your requirement. Ensure the dataset path matches the setting in `main.py`.

3. **Run the Program**:

   Execute `main.py` to train the neural network and calculate the $\eta$ value.



