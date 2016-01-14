A simple generative model of the mouse mesoscale connectome
===

All code used to generate figures is in the directory final_figures. If you run a script involving the connectivity atlas, the code will check for the .xlsx file containing the connectivity data (see below) and download it if it is not found.

## To Do
* Rename scripts to correspond to figure IDs in manuscript.
* There are still a few files we don't use, which we should get rid of.
* Remove commented out code and other functions that are never called.
* Sorry, Sid. I accidentally removed your stats scripts from the neofunkatron repo. We should put those back.
* We also need to do things like remove all references to specific environment variables. 
* Make sure that scripts that need to save intermediate files do so in a reasonable way (ideally every script should run right out of the box if someone downloads it and has all the proper dependencies).
* We also need to figure out how we're going to deal with the mouse coordinates, as I think that AIBS would prefer that we don't host any of their data on our repo. One actually quite simple way of doing this is to embed the distance matrix in 3D space using MDS. This assigns each node a 3D coordinate such that the distance matrix is essentially perfectly recovered. This will end up being equivalent to the mouse brain's true coordinates up to a rotation. I think we should definitely consider this option, especially since it's only three lines of code:
```
from sklearn import manifold
mds = manifold.MDS(n_components=3, max_iter=1000, eps=1e-10, dissimilarity='precomputed')
centroids = mds.fit_transform(distance_matrix)
```
* Figure out final color scheme. Make sure that these are all in one file and that all plotting scripts import and use it.
* Clean up all comments.
* Add test for efficiency between two versions and simplify.
* Make sure all parameters used to make models that we talk about in figures are in the same place. Right now a lot are defined in the corresponding script, rather than imported from one place, as they should be.
* Double check that code uses correct algorithms and matches the text given all the changes.
* Make sure requirements.txt contains all the right dependencies and versions, so that everything would run correctly in a virtualenv.
* 

This project uses the "linear model" published in Oh et al., (2014), which is downloadable from http://www.nature.com/nature/journal/v508/n7495/extref/nature13186-s4.xlsx .
