# Bridge Anomaly Diagnostic Pipeline using Hierarchical Transformers

An advanced structural health monitoring framework optimized for automated multi-label damage classification and post-earthquake bridge defect localization.



## 📝 Project Overview

This repository hosts a production-grade structural inspection framework designed to replace subjective manual assessments with deterministic deep learning diagnostics. Utilizing high-resolution imagery, the system targets multi-label concrete deterioration matrices where defects frequently co-exist as correlated structural pathways.

The diagnostic engine is specifically fine-tuned to map progressive degradation physics: moisture dynamics trigger foundational steel oxidation (**CorrosionStain**), generating internal expansion pressure that compromises structural integrity (**Spallation**), ultimately exposing primary load-bearing reinforcement parameters (**ExposedBars**).


## 🧬 Architectural Strategy & Engineering Foundations

The pipeline leverages deep architectural hierarchies and distinct validation loops to maintain peak precision under intense site constraints:

### 1. Feature Representation Core
*   **Backbone Framework:** Integrated **Swin Transformer Base (Swin-B)** to extract shifted window patches and map long-range global self-attention across structural boundaries.
*   **Regularization Modules:** Implemented dense drop blocks (`Dropout(0.4)`) ahead of custom linear projection heads to reduce parameter co-dependency and force model generalization.

### 2. Imbalance Resolution via Asymmetric Loss (ASL)
Standard cross-entropy optimization fails under massive dataset skews due to the dominance of clear concrete background samples over scarce target damage classes. 
*   **Asymmetric Scaling:** Implemented Asymmetric Loss (ASL) configuration setting $\gamma_{neg}=4$.
*   **Probability Clipping:** Set dynamic thresholds to $0.05$ to aggressively zero-out background noise scalars, locking 100% of gradient weight dynamic corrections strictly onto valid anomaly instances.

### 3. Explainability Engineering (SwinGradCAM)
To bypass the traditional black-box paradigm of vision models, a post-hoc **SwinGradCAM** verification mechanism intercepts the terminal spatial feature maps ($12 \times 12 \times 1024$) during an unweighted logit backward pass:
*   **Gradient Mapping:** Computes direct partial derivatives ($\frac{\partial y_c}{\partial A}$) for structural clarity.
*   **Localization Heatmap:** Bypasses saturated sigmoid layers to prevent gradient-vanishing issues, upsampling raw localized activations into clear Jet overlays right on the user dashboard.



## 📊 Evaluation & Production Metrics

*   **Macro F1-Score Verification:** Primary convergence is evaluated using unweighted Macro F1 parameters, ensuring rare safety-critical defects are scored with identical metric weight as dominant categories.
*   **Independent Activation Framework:** Replaced mutual exclusivity models (Softmax) with parallel independent **Sigmoid gates** to allow realistic multi-label outputs where individual predictions break past boundaries simultaneously.
