AdaConv
==============================

Unofficial PyTorch implementation of the Adaptive Convolution architecture for image style transfer from `"Adaptive Convolutions for Structure-Aware Style Transfer" <https://openaccess.thecvf.com/content/CVPR2021/papers/Chandran_Adaptive_Convolutions_for_Structure-Aware_Style_Transfer_CVPR_2021_paper.pdf>`__.
Disclaimer: I have not trained the model the full number of iterations yet, this is still a work in progress.

`Direct link to the adaconv module. <https://github.com/RElbers/ada-conv-pytorch/blob/master/lib/nn/adaconv/adaconv.py/>`_

`Direct link to the kernel predictor module. <https://github.com/RElbers/ada-conv-pytorch/blob/master/lib/nn/adaconv/kernel_predictor.py/>`_


Data
----

Trained with `MS COCO train2017 dataset <https://cocodataset.org>`_ for content images and `WikiArt train dataset <https://www.kaggle.com/c/painter-by-numbers>`_ for style images.
By default the content images should be placed in ``./data/MSCOCO/train2017`` and the style images in ``./data/WikiArt/train``.
You can change these directories by passing arguments when running the script: ``python .\train.py --content=<path/to/content-images> --style=<path/to/style-images>`` or by changing the default value in ``/lib/lightning/datamodule.py``


Architecture (from the original paper):
---------------------------------------

.. image:: https://raw.githubusercontent.com/RElbers/ada-conv-pytorch/master/imgs/arch_01.png

.. image:: https://raw.githubusercontent.com/RElbers/ada-conv-pytorch/master/imgs/arch_02.png


Preliminary results after training 45k iterations:

.. image:: https://raw.githubusercontent.com/RElbers/ada-conv-pytorch/master/imgs/preliminary_results.jpg

