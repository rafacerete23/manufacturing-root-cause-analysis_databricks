<img src=https://raw.githubusercontent.com/databricks-industry-solutions/.github/main/profile/solacc_logo.png width="600px">

[![DBR](https://img.shields.io/badge/DBR-17.3_LTS_ML-red?logo=databricks&style=for-the-badge)](https://docs.databricks.com/)
[![CLOUD](https://img.shields.io/badge/CLOUD-AWS_%7C_Azure-blue?style=for-the-badge)](https://databricks.com/try-databricks)

## Business Problem
Root cause analysis is critical in manufacturing for identifying and addressing the underlying factors that lead to defects, inefficiencies, and failures. By pinpointing the true sources of problems, manufacturers can implement targeted solutions to prevent recurrence, enhance product quality, reduce waste, and improve operational efficiency.

Traditional correlation-based machine learning techniques, which many companies rely on, face significant limitations in root cause analysis due to their inability to capture causality. These methods often fail to differentiate true root causes from symptoms, oversimplify complex manufacturing processes into tabular data, and neglect the manufacturing process flow. As a result, they risk producing misleading conclusions.

Causal machine learning addresses these challenges by modeling cause-and-effect relationships within complex production processes, moving beyond simple correlations. Leveraging domain knowledge, often represented as knowledge graphs, it captures causal relationships among key variables in manufacturing. This approach enables a clearer distinction between root causes and symptoms, allowing for more accurate identification of issues and their origins.

In this series of notebooks, we demonstrate how causal machine learning techniques can be applied to perform root cause analysis in manufacturing. Using a fictitious scenario where we manage a production line, we explore how various factors affect the quality of processed products, providing a practical introduction to this powerful technique.

## Run the causal core locally (no Databricks)

`local/smoke_test.py` runs the part of this accelerator that does not need Spark, MLflow or Unity Catalog: it reuses `generate_data()` from `99_utils.ipynb` and the causal graph from `01_causal_graph.ipynb`, fits the structural causal model with `dowhy`, and checks both analyses the notebooks teach.

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r local/requirements-local.txt
python local/smoke_test.py          # exact tests, about 90 s
python local/smoke_test.py --fast   # about 30 s, see below
```

`--fast` replaces the kernel independence test that decides which mechanisms changed (57 of the 70 s spent in `distribution_change`, measured with cProfile) with its approximate random-Fourier-feature version, and uses 1,000 instead of 3,000 samples for anomaly attribution. Same root cause, 3x faster, but the approximate test also flags one or two unchanged mechanisms as changed: use it while iterating, and the exact mode for conclusions.

Expected: a 9.1% defect rate in training data, rising to 13.6% after the worker mix shifts from 75/25 to 25/75, with the change attributed to `worker` (the only mechanism the generator changes) - the script asserts this.

**Graphviz is now optional.** The notebooks used to start with `%sh apt-get install graphviz`, which cannot run on serverless compute (including Databricks Free Edition) and stopped "Run all". That cell is now commented out and `pygraphviz` is out of `requirements.txt`; without it, `dowhy` draws the same graph with networkx. Running the full notebooks on Free Edition is untested: they still expect Unity Catalog permissions to create a catalog and a 17.3 LTS ML runtime.

## Reference Architecture

<img src='https://github.com/ryuta-yoshimatsu/root-cause-analysis/blob/main/images/manufacturing-process-A.png' width=800>

## Authors

<ryuta.yoshimatsu@databricks.com>, <homayoon.moradi@databricks.com>

## Project support 

Please note the code in this project is provided for your exploration only, and are not formally supported by Databricks with Service Level Agreements (SLAs). They are provided AS-IS and we do not make any guarantees of any kind. Please do not submit a support ticket relating to any issues arising from the use of these projects. The source in this project is provided subject to the Databricks [License](./LICENSE.md). All included or referenced third party libraries are subject to the licenses set forth below.

Any issues discovered through the use of this project should be filed as GitHub Issues on the Repo. They will be reviewed as time permits, but there are no formal SLAs for support. 

## License

&copy; 2025 Databricks, Inc. All rights reserved. The source in this notebook is provided subject to the Databricks License [https://databricks.com/db-license-source].  All included or referenced third party libraries are subject to the licenses set forth below.

| library                                | description             | license    | source                                              |
|----------------------------------------|-------------------------|------------|-----------------------------------------------------|
| Graphviz | An open source graph visualization software | Common Public License Version 1.0 | https://graphviz.org/download/
| pygraphviz | A Python interface to the Graphviz graph layout and visualization package | BSD | https://pypi.org/project/pygraphviz/
| networkx | A Python package for the creation, manipulation, and study of the structure, dynamics, and functions of complex networks. | BSD | https://pypi.org/project/networkx/
| dowhy | A Python library for causal inference that supports explicit modeling and testing of causal assumptions | MIT | https://pypi.org/project/dowhy/
| causal-learn | A python package for causal discovery that implements both classical and state-of-the-art causal discovery algorithms, which is a Python translation and extension of Tetrad. | MIT | https://pypi.org/project/causal-learn/
| lime | Local Interpretable Model-Agnostic Explanations for machine learning classifiers | BSD | https://pypi.org/project/lime/
| shap | A unified approach to explain the output of any machine learning model | MIT | https://pypi.org/project/shap/
