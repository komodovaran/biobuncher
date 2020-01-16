### Setup (tested on linux only!)
1. Install conda and a conda environment ("what?" "how?" - Google it!)
2. Install Tensorflow with `conda install tensorflow-gpu`. This **must** be installed as the first package. The contents
here are only tested with version 2.0, but it should work on later ones as well. If done correctly, check 
`/tests/test_tensorflow_gpu_is_working.py`
3. Install everything else with `pip install requirements.txt -r`

### Interactive scripts
Large parts run interactively in the Python-package [Streamlit](www.streamlit.io). If a script has `st_` in front of the
name, it must be run interactively through Streamlit. To launch these from the terminal, write `streamlit run st_myscript.py`

### Naming convention

* dir: relative directory (`models/mymodel/`)

* path: relative path to file (`models/mymodel/model_005.h5`)

* name: filename (`model_005.h5`)


### Troubleshooting
What to do if Streamlit doesn't finish running:

1. Hit `Ctrl+Z` in the terminal
2. If the above doesn't work, write `killall streamlit`

If, after force-quitting Streamlit you get an error like
````
tensorflow/core/kernels/cudnn_rnn_ops.cc:1624] Check failed: stream->parent()->GetRnnAlgorithms(&algorithms)
````
It means that Streamlit was improperly exited and a Tensorflow GPU session is still active.
To fix this, open a terminal and write
1. `nvidia-smi`, and search for the `pid`of the Streamlit process.
2. `kill -9 pid`, where `pid` is the number above


### Still having problems?
Open an issue and I'll see what I can do...