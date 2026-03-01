# Illustrative Code for ICLR 2026 Paper (Poster #10007190)

This repository contains the **illustrative code and data** accompanying our ICLR 2026 paper (Poster #10007190).  
It provides the minimal reproducible components needed to generate **Figure 1(a)** in the camera-ready version of the paper, as well as the main processed dataset used in our experiments.

All files in this repository are self-contained and can be executed without access to the full HealthBench raw data.

---

## 📁 Repository Contents

### **1. `1_PC50_varstd.csv`**  
This is the **main experiment dataset** used in the paper.  
It contains, for each HealthBench question:

- The **top 50 principal component (PC) scores** of its text embedding;  
- The corresponding **gold-standard–based quality gain**;  
- The corresponding **preference-based quality gain**.

These features are used as the basis of our PC-based simulation and regression analysis.

---

### **2. `PC50_varstd_RF_100G.csv`**  
This file contains the data used for the **simulation displayed in Figure 1(a)** of the paper.  
Specifically, it includes outputs from a random-forest–based simulation using the PC-50 representations.

---

### **3. `PC50_varstd_RF_100G_individual_runs.csv`**  
This file stores the **individual simulation runs** corresponding to the aggregated experiment above.  
It is used to visualize the variability and uncertainty shown in Figure 1(a).

---

### **4. `generate_plot.py`**  
A standalone Python script for generating **Figure 1(a)** exactly as it appears in the camera-ready paper.

Run:

```bash
python generate_plot.py
