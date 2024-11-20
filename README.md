# Reproducing Results from _"Diagnostic Reasoning Prompts Reveal the Potential for Large Language Model Interpretability in Medicine"_

This repository contains code and resources to reproduce the experiments from the paper _"Diagnostic Reasoning Prompts Reveal the Potential for Large Language Model Interpretability in Medicine"_ by Thomas Savage et al., published in **npj Digital Medicine (2024)**. The study evaluates whether large language models (LLMs) like GPT-4 can effectively mimic clinical reasoning processes, offering interpretable outputs for medical diagnostics.

**Paper Link:** [Nature Digital Medicine](https://www.nature.com/articles/s41746-024-01010-1)  
**Dataset:** [Google Drive Link](https://drive.google.com/drive/u/1/folders/1mDQUZ4RhyROSEycVFN_c4uyP36oyMRSe)

---

## Overview

This project aims to:

1. Explore LLM performance on clinical diagnostic reasoning using different prompting techniques.
2. Compare traditional chain-of-thought (CoT) prompting with diagnostic reasoning prompts such as differential diagnosis, analytical reasoning, intuitive reasoning, and Bayesian inference.
3. Evaluate LLM responses to modified MedQA USMLE datasets and NEJM case reports.

---

## Contents

`explore_data.ipynb` Looks at the data in the MedQA dataset and does a small amount of preprocessing.

`prompt_engineering.ipynb` Explores a simple evaluation strategy and is a good place to start experimenting with different prompts

---

## Setup Instructions

### Prerequisites

- Python >= 3.10
- Recommended: Virtual environment using `venv`.
- ```bash
    python -m venv venv
    source venv/bin/activate
  ```
- Install the required Python dependencies:
  ```bash
  pip install -r requirements.txt
  ```

### Data

The original data is from the study and can be found at [Google Drive link](https://drive.google.com/drive/u/1/folders/1mDQUZ4RhyROSEycVFN_c4uyP36oyMRSe).

## The data directory however has the cleaned and properly encoded datasets

## References

A ton of the code in `lib` is forked from the Kotaemon project at https://github.com/Cinnamon/kotaemon/

- Savage, T., Nayak, A., Gallo, R., Rangan, E., & Chen, J. H. (2024). Diagnostic reasoning prompts reveal the potential for large language model interpretability in medicine. _npj Digital Medicine, 7_, 20. [DOI:10.1038/s41746-024-01010-1](https://doi.org/10.1038/s41746-024-01010-1)

---

## Contributions

Feel free to open issues or submit pull requests to improve this project.
