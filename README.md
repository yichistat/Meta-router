# Illustrative Code for "Meta-Router: Bridging Gold-standard and Preference-based Evaluations in LLM Routing"
Paper link: https://iclr.cc/virtual/2026/poster/10007190

This repository contains the illustrative code used to generate Figure 1(a) for the main experiment of the paper.

---

## 📁 Repository Contents

### **1. `1_PC50_varstd.csv`**  
This is the dataset used for the main experiment in the paper. It is based on the HealthBench benchmark available at: https://huggingface.co/datasets/openai/healthbench.
It contains, for HealthBench questions:

- The top 50 principal component (PC) scores of its text embedding;  
- The corresponding gold-standard–based quality gain;  
- The corresponding preference-based quality gain.


---

### **2. `PC50_varstd_RF_100G.py`**  
This file contains  the simulation code corresponding to Figure 1(a) in the main paper.  

---

### **3. `PC50_varstd_RF_100G_individual_runs.csv`**  
This file stores the simulation results corresponding to the experiment above.  

---

### **4. `generate_plot.py`**  
A Python script for generating **Figure 1(a)** based on the simulation results in PC50_varstd_RF_100G_individual_runs.csv.

