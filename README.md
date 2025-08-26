## Replicating the Russell 1000 Index with Top 25 Constituents

This project replicates the Russell 1000 Index using only its top 25 constituents.
It implements several portfolio construction methods and compares them against a market-cap weighting benchmark.

Key result: Ridge Regression outperforms the benchmark with a tracking error of 2.95% and R² of 0.96 during the test period ￼.

⸻

Repository Contents

r1k_replication/ \
├── r1k_index_tracker.py       \
└── r1k_replication_report.pdf  

### Usage

Run the replication script: \
python r1k_index_tracker.py

By default, the script: \
	•	Downloads historical prices (Yahoo Finance, top 25 constituents) \
	•	Loads Fama-French factors \
	•	Fits multiple portfolio weighting strategies \
	•	Evaluates performance on training and test windows \
	•	Outputs tracking error and R² metrics

### Tech Stack 
	•	Python 3.10+ \
	•	Libraries: yfinance, pandas, numpy, scikit-learn, pandas-datareader, matplotlib
