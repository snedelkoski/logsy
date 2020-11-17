# Logsy


This is the code for the paper accepted at ICDM 2020: "Self-Attentive Classification-Based Anomaly Detection in Unstructured Logs"


For running the methods first the data must be downloaded:
https://www.usenix.org/cfdr-data (Spirit, Blue Gen\L, Thunderbird, and RAS log) and placed into /data

/code/networks.py - Contains the model classes
/code/utils.py - Contains the tokenizer, and other helper methods
/code/trainer.py - Contains the loss function and the helper methods for training and testing

The method could be run via the demo notebook script (easy to follow).

The official implementation of DeepLog used for the evaluation could be found at https://github.com/wuyifan18/DeepLog.


If you find the code or the paper useful, please consider citing:

<pre><code>@article{nedelkoski2020self,
  title={Self-Attentive Classification-Based Anomaly Detection in Unstructured Logs},
  author={Nedelkoski, Sasho and Bogatinovski, Jasmin and Acker, Alexander and Cardoso, Jorge and Kao, Odej},
  journal={arXiv preprint arXiv:2008.09340},
  year={2020}
}
</code></pre>
