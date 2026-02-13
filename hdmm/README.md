# HDMM expierment
This part of the code is demostrating HDMM experiments in our paper for table 1,2,3,4,11, 12,13,14,15 . 


In our paper we use the source code from [HDMM: The High-Dimensional Matrix Mechanism](https://github.com/dpcomp-org/hdmm), to run the HDMM experiments in tables listed above. 

# Usage
Example: to run table 1's results:
```bash
Python /hdmm/experiments/RMSE/experiment_table1_HDMM.py
```

Store the RMSE results of table 1,2,4,10,12,14,15
```bash
 /hdmm/experiments/RMSE/
```
Store the Maxvar results of table 4,11,13
```bash
 /hdmm/experiments/RMSE/
```

# Setup

Setup instructions are for an Ubuntu system.  First clone the repository, and add it to your PYTHONPATH by adding the following line to your .bashrc file:

```bash
export PYTHONPATH=/path/to/hdmm/src/
```

(Optional) now create a python virtual environment for HDMM as follows

```bash
$ mkdir ~/virtualenvs
$ python3 -m venv ~/virtualenvs/hdmm
$ source ~/virtualenvs/hdmm/bin/activate
```

And install the required dependencies:

```bash
$ pip install -r requirements.txt
```

Finally make sure the tests are passing

```bash
$ cd test/
$ nosetests
...................
----------------------------------------------------------------------
Ran 19 tests in 1.272s

OK
```

