# Model Mime
M2 (Model Mime) is a generator that, given a dataset of models (that conform a meta-model) and a set of addition edit operations, generates models that are similar to the dataset under consideration.

## Requirements 🛠
This generator has been constructed using Python. Thus, you need Python 3.X and install the requirements listed in this `requirements.txt`. I recommend you first generate a virtual environment and then install the requirements.

```
python3 -m venv <m2_env>
source <m2_env>/bin/activate
pip install -r requirements.txt
```

**Note:** The generator uses PyTorch and PyTorch Geometric. The version that is presented in `requirements.txt` is `torch-1.11.0+cu102`. Feel free to use a more suitable version.

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
