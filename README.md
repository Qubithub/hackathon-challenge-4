# Challenge 4 — Hybrid Quantum Neural Networks with CUDA-Q for Real-World Data

(Cancer Imaging or Financial Risk)

In this challenge, you will design and train a Hybrid Quantum Neural Network (HQNN) using CUDA-Q, applied to a real-world supervised learning problem.
You may choose one of two application domains:

1.	Cancer image detection using public datasets from The Cancer Imaging Archive (TCIA)

👉 https://www.cancerimagingarchive.net/

2.	Financial risk analysis using World Bank Microdata

👉 https://microdata.worldbank.org/index.php/catalog/5859 to define a financial risk prediction problem specifically centered on Mexico.

Examples of Mexico-focused tasks:

*	Predict likelihood of loan default in Mexican households.
*	Classify household financial vulnerability using socioeconomic indicators.
*	Determine high-risk vs. low-risk borrowers from Mexico-specific subsamples.
*	Identify risk clusters in Mexican microfinance or survey populations.

Your dataset selection, preprocessing steps, and final model must clearly show:

*	Extraction of Mexico-only records,
*	Feature engineering relevant to Mexican economic indicators,
*	Interpretation grounded in the Mexican socioeconomic context (e.g., income distribution, employment status, regional factors).

🌍 Motivation

Hybrid quantum–classical models are a promising near-term path for applying quantum computing to real problems, by combining:
*	Classical deep learning for rich feature extraction, and
*	Parameterized quantum circuits as trainable layers or decision heads.

This challenge asks you to demonstrate that pipeline end-to-end on real, open datasets in either healthcare or financial risk—two high-impact domains where better prediction can directly support better decisions.

📘 Scenario

Your team will:

1.	Select one of the following tracks:
* Track A – Cancer Imaging (Computer Vision)

Use a collection from TCIA and define a classification task, e.g.:

*	Tumor vs. non-tumor
*	Benign vs. malignant
* Low-risk vs. high-risk lesion subtype

*	Track B – Financial Risk (Tabular / Time Series)

Use the World Bank Microdata to define a risk-related prediction task, e.g.:

* Default vs. non-default
*	High vs. low credit risk
*	Inclusion in a vulnerable risk group
  
2.	Build a Hybrid Quantum Neural Network in CUDA-Q where:

*	A classical component (e.g., CNN, MLP, or feature extractor) processes the raw data.
*	A quantum component (parameterized quantum circuit implemented in CUDA-Q) is integrated as:

    *	A trainable quantum layer inside a neural network, or
    *	A quantum decision head operating on classical feature vectors.

3.	Train and evaluate the hybrid model, and compare it against a purely classical baseline.

🎯 Main Objective

Design, implement, and train a Hybrid Quantum Neural Network with CUDA-Q that:

*	Uses CUDA-Q to define and run a parameterized quantum circuit (variational quantum layer).
*	Integrates the quantum part into a neural network training pipeline (hybrid model).
*	Solves a supervised classification task in either:

    *	Cancer imaging (from TCIA), or
    *	Financial risk (from World Bank Microdata).

*	Achieves competitive performance vs. a classical baseline, and provides a clear analysis of the quantum contribution.

🔢 Input Data & Task Definition

You are responsible for:

1.	Selecting a specific dataset within your chosen domain:
    *	TCIA (for imaging tasks)
    *	World Bank Microdata (for risk tasks)
2.	Defining a concrete supervised learning problem, including:
*	Input type:
    *	Track A (Imaging):
    *	Raw images or preprocessed image features (e.g. embeddings from a CNN).
    *	Track B (Risk):
    *	Tabular features (e.g. demographics, income, loan characteristics, survey responses), optionally time-aggregated.
    *	Output labels (binary or multiclass).
    *	Train/validation/test splits.
3.	Preprocessing & feature engineering:
    *	Handle missing values, normalization, and any required filtering.
    *	Optionally perform dimensionality reduction (PCA, autoencoders, CNN feature extraction) before feeding features into the quantum circuit.

⚛️ Hybrid Quantum Neural Network Requirements (CUDA-Q)

Your solution must:

*	Use CUDA-Q to:
    *	Define a parameterized quantum circuit (variational circuit / quantum layer).
    *	Integrate that circuit into a gradient-based training loop (hybrid with classical optimizer).
*	Implement at least one of the following architectures:
    *	Classical feature extractor → Quantum layer → Classical output layer.
    *	Classical NN with one or more quantum layers embedded.
    *	Parallel ensemble: classical model and quantum model combined at the decision stage.

You are free to choose:
*	Encoding strategy (angle encoding, amplitude encoding, feature maps, etc.).
*	Circuit ansatz (hardware-efficient, problem-inspired, layered entangling blocks, etc.).
*	Optimizer and training strategy (SGD, Adam, hybrid approaches).

📄 Deliverable 1 — HQNN + CUDA-Q Code (Required)

Provide a repository or notebook implementing a full pipeline:

*	Data loading and preprocessing.
*	Clear task definition (what you are predicting and why).
*	Definition of:
    *	Classical components (CNN/MLP or other architecture).
    *	Quantum circuit(s) and integration using CUDA-Q.
    *	Training loop (loss, optimizer, batch strategy).
    *	Evaluation:
*	At least: accuracy + one additional metric appropriate to the task:
    *	Imaging: F1-score, AUC-ROC, sensitivity/specificity.
    *	Risk: F1-score, AUC-ROC, precision/recall, confusion matrix.
    *	A classical-only baseline for comparison.
    *	Optional:
*	Visualizations (learning curves, feature distributions, circuit depth, etc.).

📄 Deliverable 2 — Technical Report (Required)

A report (PDF or Markdown), well explained including:

1.	Problem description
    *	Domain (Cancer Imaging or Financial Risk).
    *	Dataset(s) used and justification.
    *	Exact task definition and labels.
2.	Data pipeline
    *	Preprocessing steps.
    *	Train/validation/test split strategy.
    *	Any dimensionality reduction or feature engineering.
3.	Model architecture
    *	Classical components (layers, sizes, activation functions).
    *	Quantum layer design:
        *	Encoding strategy for classical features.
        *	Circuit ansatz (gates, depth, entanglement pattern).
        *	Number of qubits and parameters.

4.	CUDA-Q integration
    *	How the quantum circuit is implemented with CUDA-Q.
    *	How gradients/parameter updates are handled.
5.	Training setup & results
    *	Hyperparameters (epochs, batch size, optimizer, learning rate).
    *	Final metrics on validation/test set.
    *	Comparison: hybrid vs classical baseline.
7.	Analysis
    *	Interpretation of results (where does the hybrid model help or struggle?).
    * Limitations (data size, model depth, noise assumptions, etc.).
    *	Ideas for improvement and future work.
    *	
🎥 Deliverable 3 — Team Presentation Video (Required)
*	Length: 3–5 minutes.
*	All team members should appear at least briefly.

*	Recommended structure:
1.	Problem and dataset chosen.
2.	Overview of your hybrid architecture (classical + quantum).
3.	How you used CUDA-Q to implement the quantum component.
4.	Key results and insights (especially vs. classical baseline).
5.	Lessons learned and possible next steps.

*	Upload the video (e.g., Google Drive) and include a viewable link in the repo (README.md or video_link.txt).


[<img src="https://qbraid-static.s3.amazonaws.com/logos/Launch_on_qBraid_white.png" width="150">](https://account.qbraid.com?gitHubUrl=REPLACEWITHGITHUBURL)
