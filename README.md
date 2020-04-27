# Logsy
A log anomaly detection method named Logsy. The code implementation is part of the paper "Logsy: Self-Attentive Classification-Based Anomaly Detectionfrom System Logs"; submitted at ACM CCS 2020



For running the methods first the data must be downloaded:
https://www.usenix.org/cfdr-data (Spirit, Blue Gen\L, Thunderbird, and RAS log) and placed into /data

/code/networks.py - Contains the model classes
/code/utils.py - Contains the tokenizer, and other helper methods
/code/trainer.py - Contains the loss function and the helper methods for training and testing

The method could be run via the demo notebook script (easy to follow).



The official implementation of DeepLog used for the evaluation could be found at https://github.com/wuyifan18/DeepLog.

