# Illustrative Code for ICLR 2026 Paper 

This repository contains the illustrative code and data accompanying Meta-Router: Bridging Gold-standard and Preference-based Evaluations in LLM Routing. It provides the minimal reproducible components needed to generate Figure 1(a) for the main experiment of the paper.

---

## 📁 Repository Contents

### **1. `1_PC50_varstd.csv`**  
This is the dataset used for the main experiment in the paper. It is based on the HealthBench benchmark available at: https://huggingface.co/datasets/openai/healthbench.
It contains, for HealthBench questions:

- The top 50 principal component (PC) scores of its text embedding;  
- The corresponding gold-standard–based quality gain;  
- The corresponding preference-based quality gain.

These features are used as the basis of our PC-based simulation and regression analysis.

---

### **2. `PC50_varstd_RF_100G.csv`**  
This file contains the data used for the simulation displayed in Figure 1(a) of the paper.  

---

### **3. `PC50_varstd_RF_100G_individual_runs.csv`**  
This file stores the simulation results corresponding to the experiment above.  

---

### **4. `generate_plot.py`**  
A standalone Python script for generating **Figure 1(a)** based on the simulation results as it appears in the paper.

Run:

```bash
python generate_plot.py
