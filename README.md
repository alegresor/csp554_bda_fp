# CSP 554: Big Data Technologies ~ Final Project
# Big Data Machine Learning

## Team Members
- Aleksei Sorokin. [asorokin@hawk.iit.edu](mailto:asorokin@hawk.iit.edu)
- Harshit Paliwal. [hpaliwal1@hawk.iit.edu](mailto:hpaliwal1@hawk.iit.edu)
- Sunny Chou. [schou2@hawk.iit.edu](mailto:schou2@hawk.iit.edu)
- Vaibhav Ramesh Kunkerkar. [vkunkerkar@hawk.iit.edu](mailto:vkunkerkar@hawk.iit.edu)

---

## Setup 
Make sure `csp554_bdt_fp/` is on you python path, this is easiest with a virtual environment.\
Run the command `pip install -r requirements.txt` to install necessary python packages. 

---

## Files
`data/` 
- `iris/` iris data
  - [UCI Machine Learning Repository](http://archive.ics.uci.edu/ml/datasets/iris)
- `turtles/` turtles data
  - [DATA.GOV turtles dataset homepage](https://catalog.data.gov/dataset/sea-turtle-population-study-in-the-coastal-waters-of-north-carolina-from-1988-06-07-to-2015-09-)
  - [National Centers for Environmental Information turtles dataset download page](https://www.nodc.noaa.gov/cgi-bin/OAS/prd/accession/download/162846)
- `wine/` wine data
  - [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/wine+quality)
- `telco/` Telco data
  - [Kaggle: Telco Customer Churn](https://www.kaggle.com/blastchar/telco-customer-churn/data)
- `auto` automotive data
  - [UCI Machine Learning Repository](http://archive.ics.uci.edu/ml/datasets/Auto+MPG)
- `housing` Boston housing data
  - [UCI Machine Learning Repository](http://archive.ics.uci.edu/ml/machine-learning-databases/housing/)

`spark_pkg/`
- `util/` utility functions for `sql.py` and `mllib.py` files
- `*/sql.py` use sql to clean dataset
- `*/mllib.py` use mllib to create classification or regression models
- `*/logs/` output logs from *.py files
- `metrics/` output metrics from models
- `out/` output figures for this package

`scikit_learn_pkg` & `r_pkg`
- `*.py` clean and model dataset
- `metrics/` output metrics from models
- `out/` output figures for this package

`explore_model_metrics.py` explore metrics output from various datasets and models

`mklogs.sh` make logs for all `sql.py` and `mllib.py` files for all datasets
  - give file permission with command:  `chmod +x spark_pkg/mklogs.sh`