**Multivariate Comparisons (MVComp)**
=================================

.. module:: mvcomp
    :synopsis: A set of functions to compute Mahalanobis D (D**2) on multimodal MRI images in a group of individuals in comparison to a reference group or within a single individual.

.. automodule:: mvcomp
    :members: neuralabc, steelec, zklsmr, stremblay18
    :undoc-members:
    :show-inheritance:


A set of functions to compute the Mahalanobis Distance (D\ :sup:`2`) 
on MRI images in a group of individuals in comparison to a reference group or within a single individual.

Tha Mahalanobis Distance (D\ :sup:`2`) is a measure of the distance between a point P (i.e. an MRI feature in a voxel or an ROI) 
and a distribution D (i.e. the distribution of the same MRI feature in a reference group/another ROI). 
The Mahalanobis Distance is a generalization of the Euclidean Distance that takes into account the covariance of the data, which is important given 
the high correlation between MRI-derived microstructural measures. 

For a detailed description of how D\ :sup:`2` can applied to neuroimaging data, please refer to the our paper:

    .. code-block:: latex 

        Tremblay, SA, Alasmar, Z, Pirhadi, A, Carbonell, F, Iturria-Medina, Y, Gauthier, C, Steele, CJ, (2024). MVComp toolbox: MultiVariate Comparisons of brain MRI features accounting for common information across metrics.
        BioRxiv, https://doi.org/10.1101/2024.02.27.582381.

=================================
--------------
**Installation**
--------------
To install `mvcomp`, you can clone the repository from GitHub and add the path to the package in your python script:

to clone: 

.. code-block:: bash

    git clone https://github.com/neuralabc/mvcomp.git

and in your python script:

.. code-block:: python

    import sys
    sys.path.append('path/to/mvcomp')
    import mvcomp



**Quick Start**
--------------

Here's an example on how to use `mvcomp` if your data is ready:

.. code-block:: python

    from mvcomp import model_comp

    result = model_comp(feature_in_dir, model_dir=None, suffix_name_comp=".nii.gz", exclude_comp_from_mean_cov=True,
                        suffix_name_model=".nii.gz", mask_f=None, mask_img=None, verbosity=1,
                        mask_threshold=0.9, subject_ids=None, exclude_subject_ids=None, feat_sub=[], return_raw=False)

    print(result)

Here, `feature_in_dir` is the path to the directory containing subject subdirectories with the MRI features. 
You can set `exclude_comp_from_mean_cov` to implement the leave-one-subject-out, or provide a `model_dir` to compare the features in `feature_in_dir` to the features in `model_dir` (which serves as the reference.



**Reference Us!**
---------------------
if you use the MVComp package in your research, please cite the following paper:

    .. code-block:: latex 

        Tremblay, SA, Alasmar, Z, Pirhadi, A, Carbonell, F, Iturria-Medina, Y, Gauthier, C, Steele, CJ, (2024). MVComp toolbox: MultiVariate Comparisons of brain MRI features accounting for common information across metrics.
        BioRxiv, https://doi.org/10.1101/2024.02.27.582381.


**Having issues or need help?** 
-------------------------------
if you have any issues or need help with the MVComp package, please open an issue on the GitHub repository:
    github.com/neuralabc/mvcomp/issues

or email us at:
    zaki.alasmar@mail.concordia.ca

**Contribute**  
--------------
if you would like to contribute to the MVComp package, please open an issue or a pull request on the GitHub repository:
    https://github.com/neuralabc/mvcomp.git

---------------------
=====================


