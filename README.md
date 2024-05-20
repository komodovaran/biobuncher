LSTM Variational Autoencoders for Time Series Data

This repository contains implementations of LSTM-based autoencoders and bidirectional LSTM VAEs designed to encode variable-length time series into latent spaces. These latent representations can be used for tasks such as clustering, anomaly detection, and feature extraction. The models leverage TensorFlow and TensorFlow Probability for robust and scalable neural network architectures.

Note that this code has never been published, or tested extensively, and is provided as-is.

### What does it do?

Uses LSTM VAE to easily categorize data with absolute minimal knowledge about the data.

<img width="679" alt="Screenshot 2024-05-20 at 16 33 50" src="https://github.com/komodovaran/biobuncher/assets/20357875/4f2f6ccd-5fc4-4143-aeaf-fb557da207f1">

### Setup (tested on linux only!)
1. Install conda and a conda environment. Conda installation instructions for
Linux can be found on the website, as well as how to create an environment.
2. Install Tensorflow with `conda install tensorflow-gpu=2.0.0`. This **must**
be installed as the first package. The contents here are only tested with 
version 2.0, but it should work on later ones as well. If done correctly,
check the script at  `/checks/test_tensorflow_gpu_is_working.py`
3. Install the rest of the conda requirements with   
````
conda install -f -y -q --namepy37 -c conda-forge --file conda_requirements.txt
````
4. Install everything else with `pip install -r requirements.txt`
5. If Tensorflow is installed correctly, run 
`checks/test_tensorflow_gpu_is_working`. If the device is correctly set up,
Tensorflow is working and you're good to go!
6. Conda and pip don't talk together, and this breaks some of the package
installations. If for some reason a package was not installed, try running a 
script until you hit a `ModuleNotFound: no module named name_of_package` error,
and try installing the module with `pip install name_of_package`.

### Interactive scripts
Large parts run interactively in the Python-package
[Streamlit](www.streamlit.io). If a script has `st_` in front of the name, it
must be run interactively through Streamlit (or else it doesn't produce any
visible output). To launch these from the terminal, write 
`streamlit run st_myscript.py`

### Naming convention

* dir: relative directory (`models/mymodel/`)

* path: relative path to file (`models/mymodel/model_005.h5`)

* name: filename (`model_005.h5`)

* To access something one directory up, write `../` in front of the directory
name. Two directories up is `../../`, and so on.

* All paths in use are defined in `lib/globals.py`, so they can be conveniently
changed once here, rather than everywhere in the code.

### Data format
Every dataset is preprocessed so that eventually you'll have a `hdf` file and a
corresponding `npz` file. All computations are done on the `npz` file because
it's much faster, and compatible with tensorflow. However, group order must be
preserved according to the parent `hdf` dataframe.

A `npz` has just a single group index (i.e. '512' means trace id 512 (remember,
Python counts from 0!), whereas a dataframe may have both an `id` and `sub_id`
if it's combined from multiple sources. In that case, the `id` will correspond
to the `npz` index (i.e. the order of appearance), and `sub_id` will be the
actual group index in the sub-dataset (which is currently not used). Group order
is only preserved if dataframe groups are sorted by `['file', 'particle']`, or
for combined dataframes `['source', 'file', 'particle']`. To combine dataframes,
the inputs are stacked in loaded order (which must therefore also be sorted!).
All of this is done automatically, if the right sequence of steps is taken. 


### Scripts to run, step by step
1. `get_cme_tracks.py` to convert from CME `.mat` files to a dataframe.
2. `prepare_data.py` to filter out too short data (set it low initially if you 
want to be safe - The model will work almost equally well, regardless of the 
minimum length). If desired, can also remove tracks that would be cut by the
tracking start/end (i.e. if something starts at frame 0 of the video, it's
removed, because you can't be sure if the actual event started at "frame -10".).
This can also be disabled if not desirable/applicable for the data at hand. 
 
3. `train_autoencoder.py` to train a model on the data.
4. `st_predict.py` to predict and plot the data. Initially, a UMAP model is
trained. This takes a while. It might even time out your Streamlit session, but
don't touch anything and it'll be ready eventually.
5. Every cluster is saved as a combination of model + data names, and will be
output to `results/cluster_indices/`. This contains the indices of every trace
(see above on how indexing works), and which cluster they belong to. Note that
every change in the analysis **OVERWRITES** the automatically created file
containing cluster indices. If you have reached a point where you want to save
them, go to `results/cluster_indices/` and rename the file so you're sure it
won't be overwritten. 
6. `st_eval.py` once clustering is done and you want to explore the data. It
currently doesn't have much functionality. Only looking at one/more specific
datasets/clusters...

### Things to avoid
In order to preserve group ordering, the original dataframes must be run through
 `prepare_data.py` if they need to be filtered in some way. **DO NOT** run a
 combine dataframe through a filter, because this messes up the internal group
 ordering that was first established when creating the combined dataframe.

### Troubleshooting
#### Packages are missing
If any scripts raise complaints about packages I may have missed, they can be installed with
`pip install packagename`


#### The interface is slow!
Streamlit was never designed for super heavy computations. The underlying calculations are as fast as possible
but due to the way Streamlit is set up, it appears to be slow. Rest assured, after you put in the parameters,
Streamlit will get there eventually. Just don't touch anything until it's done, because the script will re-run
whenever any parameters are changed.


#### What to do if Streamlit doesn't finish running:

1. Hit `Ctrl+Z` in the terminal
2. If the above doesn't work, write `killall streamlit`

If after using Streamlit you get an error like
````
tensorflow/core/kernels/cudnn_rnn_ops.cc:1624] Check failed: stream->parent()->GetRnnAlgorithms(&algorithms)
````
It means that a Tensorflow GPU session is still active from the Streamlit session.
To fix this, open a terminal and write
1. `nvidia-smi`, and search for the `pid`of the Streamlit process.
2. `kill -9 pid`, where `pid` is the number above

#### What to do if you can't delete a directory:
Tensorflow by default creates directories with incorrect permissions for PyCharm.
To fix this and make them deletable from PyCharm navigate to the base directory and write
`sudo chmod -R 777 models`.

### Still having problems?
Open an issue and I'll see what I can do...
