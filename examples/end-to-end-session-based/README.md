# End-to-end session-based recommendation

These end-to-end example notebook focuses on the following:

* Preprocessing the Yoochoose e-commerce dataset.
* Generating session features with on GPU.
* Using the NVTabular dataloader with the Pytorch.
* Training a session-based recommendation model with a Transformer architecture (XLNET).
* Exporting the preprocessing workflow and trained model to Triton Inference Server (TIS).
* Sending request to TIS and generating next-item predictions for each session.

Refer to the following notebook:

* [Session-based recommendation with Yoochoose](end-to-end-session-based-with-Yoochoose.ipynb)