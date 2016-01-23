A simple generative model (AKA Neofunkatron) of the mouse mesoscale connectome
===

All code used to generate figures is in the directory final_figures. If you run a script involving the connectivity atlas, the code will check for the .xlsx file containing the connectivity data (see below) and download it if it is not found.

## To Do
* Rename scripts to correspond to figure IDs in manuscript.
* There are still a few files we don't use, which we should get rid of.
* Remove commented out code and other functions that are never called.
* We also need to do things like remove all references to specific environment variables. 
* Make sure that scripts that need to save intermediate files do so in a reasonable way (ideally every script should run right out of the box if someone downloads it and has all the proper dependencies).
* Figure out final color scheme. Make sure that these are all in one file and that all plotting scripts import and use it.
* Clean up all comments.
* Add test for efficiency between two versions and simplify.
* Make sure all parameters used to make models that we talk about in figures are in the same place. Right now a lot are defined in the corresponding script, rather than imported from one place, as they should be.
* Double check that code uses correct algorithms and matches the text given all the changes.
* Make sure requirements.txt contains all the right dependencies and versions, so that everything would run correctly in a virtualenv.

This project uses the "linear model" and brain region distance matrix used in Oh et al., (2014). Both are available in the supplementary information of http://www.nature.com/nature/journal/v508/n7495/full/nature13186.html.
