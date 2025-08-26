This project replicates the Russell 1000 Index using only its top 25 constituents.
It implements several portfolio construction methods and compares them against a market-cap weighting benchmark.

Key result: Ridge Regression outperforms the benchmark with a tracking error of 2.95% and R² of 0.96 during the test period ￼.

⸻

Repository Contents

r1k_replication/
├── r1k_index_tracker.py       # Main script with implementation
└── r1k_replication_report.pdf # Full research report

⚙️ Installation

Clone the repository:
git clone https://github.com/yourusername/r1k_replication.git
cd r1k_replication

Create a virtual environment and install dependencies:
python -m venv .venv
source .venv/bin/activate   # Linux/Mac
.venv\Scripts\activate      # Windows

pip install -r requirements.txt

Usage

Run the replication script:
python r1k_index_tracker.py

By default, the script:
	•	Downloads historical prices (Yahoo Finance, top 25 constituents)
	•	Loads Fama-French factors
	•	Fits multiple portfolio weighting strategies
	•	Evaluates performance on training and test windows
	•	Outputs tracking error and R² metrics

Tech Stack
	•	Python 3.10+
	•	Libraries: yfinance, pandas, numpy, scikit-learn, pandas-datareader, matplotlib
