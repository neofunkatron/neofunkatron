A simple generative model (AKA Neofunkatron) of the mouse mesoscale connectome
===

All code used to generate figures is in the directory final_figures. If you run a script involving the connectivity atlas, the code will check for the .xlsx file containing the connectivity data (see below) and download it if it is not found.

## To Do
* Remove commented out code and other functions that are never called.
* Clean up all comments.
* Add test for efficiency between two versions and simplify.
* Make sure all parameters used to make models that we talk about in figures are in the same place. Right now a lot are defined in the corresponding script, rather than imported from one place, as they should be.
* Double check that code uses correct algorithms and matches the text given all the changes.
* Make sure requirements.txt contains all the right dependencies and versions, so that everything would run correctly in a virtualenv.

This project uses the "linear model" and brain region distance matrix used in Oh et al., (2014). Both are available in the supplementary information of http://www.nature.com/nature/journal/v508/n7495/full/nature13186.html.
