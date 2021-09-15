End-to-end session-based recommendation
=======================================

This end-to-end example notebook is focuses on:

* Preprocessing the Yoochoose e-commerce dataset
* Generating session features with on GPU
* Using the NVTabular dataloader with the Pytorch
* Training a session-based recommendation model with a Transformer architecture (XLNET)
* Exporting the preprocessing workflow and trained model to Triton Inference Server (TIS)
* Sending request to TIS and generating next-item predictions for each session

.. toctree::
   :maxdepth: 1

   End-to-end session-based with Yoochoose <end-to-end-session-based-with-Yoochoose>
