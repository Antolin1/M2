# Model Mime
M2 (Model Mime) is a generator that, given a dataset of models (that conform a meta-model) and a set of addition edit operations, generates models that are similar to the dataset under consideration.

## Requirements 🛠
This generator has been constructed using Python. 
Thus, you need Python 3.8.X and install the requirements listed in this `requirements.txt`. 
I recommend you first generate a virtual environment (with conda) and then install the requirements.

```
conda create -n <m2_env> python=3.8
conda activate <m2_env>
sudo apt-get install graphviz graphviz-dev
pip install -r requirements.txt
```

The generator uses [PyTorch](https://pytorch.org/get-started/locally/) and [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html). 
The versions that were used when developing the project were:
- `torch-1.11.0+cu102`
- `torchvision-0.12.0+cu102`
- `torchaudio-0.11.0`
- `torch-geometric-2.0.4`
- `torch-scatter-2.0.9`
- `torch-sparse-0.6.13`
- `torch-spline-conv-1.2.1`

Feel free to use a more suitable version.

## Running the generator 🚀

In the repository you can find a `main.py` script that is in charge of running everything.
To train our generator you can do the following:

```sh
python main.py --train 
    --training_dataset <training_dataset>
    --metamodel <metamodel>
    --root_object <root_object>
    --model_path <model_path>
```

- `training_dataset`: the folder where the training dataset is located.
- `metamodel`: the path to the meta-model.
- `root_object`: the root object of the meta-model (that contains everything).
- `model_path`: the folder where the trained neural network will be stored.

To generate models using the trained generator:

```sh
python main.py --inference
    --metamodel <metamodel>
    --root_object <root_object>
    --model_path <model_path>
    --max_size <max_size>
    --n_samples <n_samples>
    --output_path <output_path>
```

- `max_size`: the maximum size of the generated models.
- `n_samples`: the number of models that will be generated.
- `output_path`: the folder where the generated models will be placed.
