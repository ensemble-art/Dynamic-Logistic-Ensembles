# Dynamic Logistic Ensembles with Recursive Probability

Welcome to the **Dynamic Logistic Ensembles** repository! This project showcases a novel approach for binary classification that extends traditional logistic regression into an ensemble of models organized in a tree-like structure. The ensemble automatically partitions data to capture underlying group structures—especially useful in cases where such structures exist but are not explicitly observable through features alone. 

This repository provides:

- **Data preprocessing and augmentation** notebooks
- **Dynamic logistic ensemble** model demonstration notebooks
- **Presentation slides** detailing the approach and recursive formulations
- **Published paper** (PDF) describing the theoretical underpinnings and methodology

---

## Table of Contents

1. [Repository Overview](#repository-overview)  
2. [Project Motivation & Features](#project-motivation--features)  
3. [Folder & File Descriptions](#folder--file-descriptions)  
4. [Getting Started](#getting-started)  
   - [Prerequisites](#prerequisites)  
   - [Installation](#installation)  
   - [Usage](#usage)  
5. [Results & Visualization](#results--visualization)  
6. [Citation](#citation)  
7. [License](#license)  
8. [Contact](#contact)  

---

## Repository Overview

**Dynamic logistic ensembles** address the limitations of standard logistic regression in complex datasets that contain multiple internal clusters or subgroups. By recursively combining the outputs of multiple logistic regression nodes, these ensembles can approximate more nuanced decision boundaries without losing the interpretability characteristic of logistic models.

Key highlights:

- **Recursive Probability Calculation**  
  The ensemble’s final prediction is recursively derived. Each node splits data into “left” and “right” subsets (similar to a binary tree), and the probability of belonging to the positive class is computed by descending the tree of logistic models.

- **Analytical Gradient Derivation**  
  The cost and its gradients for each node are derived analytically, making the approach both rigorous and computationally efficient.

- **Automatic Subset Splitting**  
  Instead of manually specifying clusters, the ensemble method captures latent group structures inherently—particularly valuable in datasets where clusters are not immediately obvious.

---

## Project Motivation & Features

1. **Motivation**:  
   - Traditional logistic regression often struggles when data has multiple internal clusters that share the same label but differ in their feature distributions.  
   - Ensemble methods like boosting or bagging can sometimes obscure interpretability. Dynamic logistic ensembles retain transparent, per-node coefficients for interpretability.

2. **Features**:  
   - **Scalable** to multiple layers (tree depth) for increasingly complex data.  
   - **Interpretability**: Each node is a logistic regression model with well-defined coefficients.  
   - **Cost & Gradient** derived from first principles for the entire ensemble.  
   - **Data Augmentation**: Gaussian noise is added to simulate real-world-like subgroups in training data.

---

## Folder & File Descriptions

Below is an overview of the key files included in this repository:

1. **`allwine.csv`**  
   - A CSV file containing the augmented Wine Quality dataset. This dataset has 10 input features (e.g., acidity, pH, etc.) and a target label (`quality`). Some rows are augmented with Gaussian noise to simulate internal grouping.

2. **`Data_Augmentation.ipynb`**  
   - Jupyter Notebook demonstrating how the original Wine Quality data was **augmented** with Gaussian noise.  
   - Includes **PCA visualization** to illustrate the difference between the original vs. augmented data.

3. **`Dynamic_Ensemble_Models_Demo.ipynb`**  
   - Main notebook for **demonstrating** the dynamic logistic ensemble approach.  
   - Walks through single-layer, double-layer (2), triple-layer (3), and quadruple-layer (4) ensembles.  
   - Illustrates cost, ROC curves, and performance metrics such as accuracy, recall, precision, and AUC.

4. **`Dynamic_Logistic_Ensembles_with_Recursive_Probability_YU.pptx`**  
   - Presentation slides summarizing the conceptual approach, recursive formulation, and results.  
   - Useful for a quick high-level overview or for use in academic/industry presentations.

5. **`Dynamic Logistic Ensemble presentation.pptx`**  
   - An additional presentation deck explaining logistic regression ensembles, architectural details, and initial findings.

6. **`Dynamic_Recursive_Logistic_Ensemble_Model.pdf`**  
   - **Accepted IEEE conference paper** detailing the full methodology, mathematical derivations, and experimental results.  
   - Contains references, discussions on limitations, and suggestions for future work.

---

## Getting Started

### Prerequisites

- **Python 3.7+**  
- Common scientific libraries:  
  - `numpy`  
  - `pandas`  
  - `matplotlib`  
  - `scikit-learn`  
  - `scipy`  
  - `seaborn` (optional for enhanced visualizations)  

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/ensemble-art/Dynamic-Logistic-Ensembles.git
   cd Dynamic-Logistic-Ensembles
   ```

2. **Install dependencies** (e.g., via `pip`):
   ```bash
   pip install -r requirements.txt
   ```
   *(If a `requirements.txt` is not provided, install the libraries mentioned under [Prerequisites](#prerequisites).)*

3. **Open Jupyter Notebooks**:
   ```bash
   jupyter notebook
   ```
   Then navigate to `Data_Augmentation.ipynb` or `Dynamic_Ensemble_Models_Demo.ipynb`.

### Usage

1. **Data Augmentation**  
   - Run the cells in `Data_Augmentation.ipynb` to see how Gaussian noise is added to the original dataset.  
   - Visualize the augmented data with PCA to confirm the presence of simulated subgroups.

2. **Dynamic Logistic Ensembles**  
   - Open `Dynamic_Ensemble_Models_Demo.ipynb`.  
   - Follow the notebook cells in sequence: data loading, model initialization, training, and evaluation.  
   - Adjust **`n_layers`** in the code to explore deeper or shallower ensemble structures.

3. **Presentations & Paper**  
   - Explore the `.pptx` files for an overview of the methodology.  
   - Refer to the `Dynamic_Recursive_Logistic_Ensemble_Model.pdf` for the full academic paper detailing the approach, equations, and results.

---

## Results & Visualization

- **Accuracy, AUC, Recall, Precision** metrics are logged for each ensemble depth (1-layer up to 4-layer).  
- Plots such as cost vs. iteration and ROC curves are automatically generated in the notebooks.  
- The repository includes final converged values showing how deeper ensembles can potentially capture complex data patterns, though diminishing returns or overfitting can occur beyond a certain depth.

Example performance comparison (sample from the notebooks):

| Metric         | Baseline LR | 1-Layer | 2-Layer | 3-Layer | 4-Layer |
|----------------|------------:|--------:|--------:|--------:|--------:|
| **Train Acc**  |       0.701 |  0.7435 |  0.7576 |  0.7869 |  0.8202 |
| **Test Acc**   |       0.689 |  0.7375 |  0.7547 |  0.7641 |  0.7531 |
| **Test AUC**   |       0.754 |  0.8019 |  0.8257 |  0.8435 |  0.8320 |
| **Recall**     |       0.656 |  0.6688 |  0.6972 |  0.7224 |  0.7476 |
| **Precision**  |       0.698 |  0.7709 |  0.7837 |  0.7842 |  0.7524 |

---

## Citation

If you find this work helpful in your research or projects, please cite the **IEEE conference paper**:

```
@inproceedings{khan2024dynamic,
  title     = {Dynamic Logistic Ensembles with Recursive Probability and Automatic Subset Splitting for Enhanced Binary Classification},
  author    = {Khan, Mohammad Zubair and Li, David},
  booktitle = {2024 IEEE 15th Annual Ubiquitous Computing, Electronics \& Mobile Communication Conference (UEMCON)},
  year      = {2024},
  publisher = {IEEE},
  doi       = {10.1109/UEMCON62879.2024.10754761}
}
```

---

## License

This project is licensed under the [MIT License](LICENSE) – feel free to use or modify the code for academic and commercial purposes, while giving appropriate credit to the original authors.

---

## Contact

- **Author**: Mohammad Zubair Khan ( [mkhan10@mail.yu.edu](mailto:mkhan10@mail.yu.edu) )  
- **Co-Author**: David Li ( [david.li@yu.edu](mailto:david.li@yu.edu) )  
- **Institution**: Katz School of Science and Health, Yeshiva University, New York, NY

For questions, collaborations, or suggestions, please open an **Issue** in this repo or reach out via email.

**Thank you** for checking out this project! We hope it helps in understanding how logistic regression can be scaled to handle internal group structures while preserving interpretability. Contributions and feedback are welcome. Have fun exploring dynamic logistic ensembles!
