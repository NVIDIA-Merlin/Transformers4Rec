# End-to-end session-based recommendation

These end-to-end example notebooks focus on the following:

* Preprocessing the Yoochoose e-commerce dataset.
* Generating session features with on GPU.
* Using the NVTabular dataloader with PyTorch.
* Training a session-based recommendation model with a Transformer architecture (XLNET).
* Exporting the preprocessing workflow and trained model to Triton Inference Server (TIS).
* Sending request to TIS and generating next-item predictions for each session.

Refer to the following notebooks:

* [ETL with NVTabular](01-ETL-with-NVTabular.ipynb)
* End-to-end session-based recommendation: [PyTorch](02-End-to-end-session-based-with-Yoochoose-PyT.ipynb)