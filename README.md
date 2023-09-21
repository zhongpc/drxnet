# DRXNet
 
A Python package for learning representations of battery discharge voltage profiles across diverse compositions in disordered rocksalt cathodes (DRX).

<img src='/battery_dataset.png' class="center" width="99%">


**Inputs**:

1. **composition** (string)
2. **current density rate** (mA/g)
3. **test voltage window** (V)
4. **cycle number** (integer)

## Dependency
* Requirements: `pymatgen`, `torch`, `torch-scatter`.
* Here is a simple installation if you are using `DRXNet` for prediction on CPU:

```sh
conda create -n drxnet python=3.9
conda activate drxnet

pip install pymatgen
pip install torch==1.12.0 -f https://download.pytorch.org/whl/cpu/torch_stable.html
pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.12.0+cpu.html
```

## Installation
```sh
python setup.py install
```



## Examples

**Due to the high value of the DRX test dataset and the universal models**, the pretrained weights will not be released at this moment. We provide an example script for DRXNet training using the initial ten cycles of Li$_{1.2}$Mn$_{0.4}$Ti$_{0.4}$O$_{2.0}$ (LMTO) DRX (see `./examples/train_example/`, simply run `python train_LMTO.py`).

The model and the training scheme are highly expandable to other systems, such as element-doping in NCM materials, high-entropy Na-ion cathode, etc. The practitioners can customize their own electrochemical dataset and parameterize the model to make predictions.



These examples require the pretrained model, we apologize that we are not able to release the DRX test dataset and models now.

(1) Discharge capacity predictions in Li-Mn-O-F chemical space `./examples/[1]-predict_LMOF_chemical_space.ipynb
`

(2) Multi-rate performance predictions of high-entropy DRXs `./examples/[2]-predict_HE-DRX.ipynb`

## Quick start

If you have generated the model weights, you can save them as `./pretrained/model_0`, and you can run the following example to get the voltage profile.

```py
from drxnet.funciton import load_pretrained_models, ensemble_prediction, voltageFeaturizer
from drxnet.drxnet.model import DRXNet
from drxnet.core import Featurizer
import matplotlib.pyplot as plt

############# load the pretrained models #############
model_list = load_pretrained_models()
vol_featurizer  = voltageFeaturizer()

############# predict discharge curve #############
Q_, V_ , *_ =  ensemble_prediction(model_list = model_list,
                                   vol_featurizer = vol_featurizer,
                                   composition_string = 'Li1.2Mn0.2Cr0.2Ti0.4O2.0',
                                   input_rate = 20,
                                   input_cycle = 1,
                                   input_V_low = 1.5,
                                   input_V_high = 4.8)

plt.plot(Q_, V_, '-', color = 'k', linewidth = 2.0)
plt.show()

```

## Acknowledgement

The [Roost](https://github.com/CompRhys/roost) (Representation Learning from Stoichiometry) and [mat2vec](https://github.com/materialsintelligence/mat2vec) is used for compositional encoding in DRXNet. Please consider citing the relevant works:


* R. E. A. Goodall and A. A. Lee, Predicting Materials Properties without Crystal Structure: Deep Representation Learning from Stoichiometry, Nat. Commun. 11, 6280 (2020). [[link]](http://www.nature.com/articles/s41467-020-19964-7)

* V. Tshitoyan, J. Dagdelen, L. Weston, A. Dunn, Z. Rong, O. Kononova, K. A. Persson, G. Ceder, and A. Jain, Unsupervised Word Embeddings Capture Latent Knowledge from Materials Science Literature, Nature 571, 95 (2019). [[link]](http://www.nature.com/articles/s41586-019-1335-8)
