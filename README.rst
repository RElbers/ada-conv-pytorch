AdaConv
==============================

Unofficial PyTorch implementation of the Adaptive Convolution architecture for image style transfer from `"Adaptive Convolutions for Structure-Aware Style Transfer" <https://openaccess.thecvf.com/content/CVPR2021/papers/Chandran_Adaptive_Convolutions_for_Structure-Aware_Style_Transfer_CVPR_2021_paper.pdf>`__.
I tried to be as faithful as possible to the what the paper explains of the model, but not every training detail was in the paper so I had to make some choices regarding that.
If something was unclear I tried to do what AdaIn does instead. Results are at the bottom of this page.


`Direct link to the adaconv module. <https://github.com/RElbers/ada-conv-pytorch/blob/master/lib/adaconv/adaconv.py/>`_

`Direct link to the kernel predictor module. <https://github.com/RElbers/ada-conv-pytorch/blob/master/lib/adaconv/kernel_predictor.py/>`_

Usage
-----

The parameters in the commands below are the default parameters and can thus be omitted unless you want to use different options.
Check the help option (``-h`` or ``--help``) for more information about all parameters.
To train a new model:

.. code::

    python train.py --content ./data/MSCOCO/train2017 --style ./data/WikiArt/train


To resume training from a checkpoint (.ckpt files are saved in the log directory):

.. code::

    python train.py --checkpoint <path-to-ckpt-file>


To apply the model on a single style-content pair:

.. code::

    python stylize.py --content ./content.png --style ./style.png --output ./output.png --model ./model.ckpt


To apply the model on every style-content combination in a folder and create a table of outputs:

.. code::

    python test.py --content-dir ./test_images/content --style-dir ./test_images/style --output-dir ./test_images/output --model ./model.ckpt


Weights
=======
`Pretrained weights can be downloaded here. <https://drive.google.com/file/d/17h-Hd08n-f_5D8cDV08dpB_-W1cs5jbt/view?usp=sharing>`_
Move ``model.ckpt`` to the root directory of this project and run ``stylize.py`` or ``test.py``.
You can finetune the model further by loading it as a checkpoint and increasing the number of iterations.
To train for an additional 40k (200k - 160k) iterations:

.. code::

    python train.py --checkpoint ./model.ckpt --iterations 200000


Data
====

The model is trained with the `MS COCO train2017 dataset <https://cocodataset.org>`_ for content images and the `WikiArt train dataset <https://www.kaggle.com/c/painter-by-numbers>`_ for style images.
By default the content images should be placed in ``./data/MSCOCO/train2017`` and the style images in ``./data/WikiArt/train``.
You can change these directories by passing arguments when running the script.
The test style and content images in the ``./test_images`` folder are taken from the `official AdaIn repository <https://github.com/xunhuang1995/AdaIN-style/tree/master/input>`_.


Results
=======
Judging from the results I'm not convinced everything is as the original authors did, but without an official repository it's hard to compare implementations.
Results after training 160k iterations:

.. image:: https://raw.githubusercontent.com/RElbers/ada-conv-pytorch/master/imgs/results_table_256.jpg

Comparison with reported results in the paper:

.. image:: https://raw.githubusercontent.com/RElbers/ada-conv-pytorch/master/imgs/results_comparison.jpg


Architecture (from the original paper):
---------------------------------------

.. image:: https://raw.githubusercontent.com/RElbers/ada-conv-pytorch/master/imgs/arch_01.png

.. image:: https://raw.githubusercontent.com/RElbers/ada-conv-pytorch/master/imgs/arch_02.png

