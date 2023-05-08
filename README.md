# evolve_cart
Jupyter notebook in which students can learn to evolve the controller for the AI gym continuous cart problem.

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/guidoAI/evolve_cart.git/master)

# How to install and run locally
The easiest way to install and run the Jupyter notebook locally is to:
(1) Create a virtual environment for Python (venv) (see, e.g., [this explanation](https://docs.python.org/3/library/venv.html) for Windows ).
(2) Install the required packages with the help of the requirements.txt file (see, e.g., [here](https://stackoverflow.com/questions/7225900/how-can-i-install-packages-using-pip-according-to-the-requirements-txt-file-from) how to do that).
(3) Add the environment as a Kernel to Jupyter notebook (see, e.g., [this blog](https://janakiev.com/blog/jupyter-virtual-envs/#add-virtual-environment-to-jupyter-notebook)).

On Windows in the (GIT) command window:

```
git clone git@github.com:guidoAI/evolve_cart.git
python -m venv evolve_cart
cd evolve_cart
.\Scripts\activate.bat
pip install -r requirements.txt
pip install ipykernel
python -m ipykernel install --user --name=evolve_cart
jupyter notebook
```

Then, in the Jupyter notebook, select ``evolve_cart`` as a Kernel.

