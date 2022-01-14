Requirements and Instalaltion
==============================


Requirements
------------

Vegphenome uses MaskRCNN for detection and ResNet models for orientation correction(for serial pipeline) of vegs due to its use of deep learning it is highy recommended to use
a machine with recent GPU recommended to use >= GTX 1080 TI
Install the requirements using requirements.txt file


.. code-block:: console

   (.venv) $ git clone vegphenome
   (.venv) $ cd vegphenome && pip install -r requirements.txt


Installation
------------

To use vegphenome, first clone it, build it and install it using pip:

.. code-block:: console

   (.venv) $ python setup.py bdsit_wheel
   (.venv) $ cd dsit && pip install vegphenome*version*.whl