A simple generative model (AKA "the neofunkatron") of the mouse mesoscale connectome
===

All code used to generate figures is in the directory final_figures. To run any piece of code, clone this repository, change directory to the repository root, and call python final_figures/figXXXX.py . Figures generated will be saved as pdf and png formats in the directory from which the command was invoked. Some scripts take a long time to run (especially scripts involving statistics over many realizations of the random graphs). Some scripts also save a temporary file in the current directory in order to speed up processing later on.

All of the code for generating custom random graphs is in the random_graph directory.

This project uses the "linear model" and brain region distance matrix used in Oh et al., (2014). Both are available in the supplementary information of http://www.nature.com/nature/journal/v508/n7495/full/nature13186.html. When the code is run, these files will be downloaded to the current directory the first time they are needed. You must be connected to the internet for this to work.

If you have any questions, feel free to contact any of the owners of this repository (Henriksen, Pang, Wronkiewicz).

All code is written in Python 2.7. The versions of external libraries we used are given in requirements.txt. To run our code we recommend first installing [Anaconda](https://www.continuum.io/downloads), a popular scientific Python distribution that includes all the packages we used.
