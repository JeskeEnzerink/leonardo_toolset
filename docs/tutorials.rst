:html_theme.sidebar_secondary.remove:

Tutorials
=========
This section contains various tutorials showcasing light-sheet data processing with :mod:`Leonardo`.

Basic Tutorials
----------------
This section contains notebook tutorials showcasing how to do stripe removal using :mod:`Leonardo-DeStripe`, 
how to fuse datasets with dual-sided illumination and/or dual-sided detection using :mod:`Leonardo-Fuse`, 
and how to build an optimal workflow to resolve sample-induced aberrations using :mod:`Leonardo-DeStripe-Fuse`.

.. toctree::
    :maxdepth: 1

    tutorials/notebook/Leonardo_DeStripe
    tutorials/notebook/Leonardo_Fuse_(along_illumination)
    tutorials/notebook/Leonardo_Fuse_(along_detection)
    tutorials/notebook/Leonardo_DeStripe_Fuse

Video Tutorials for the Napari Plugins
--------------------
This section contains video tutorials showcasing how to use our Napari interface to easily realize aforementioned functions. Example dataset is downloadable at 
`zenodo <https://zenodo.org/records/14215090>`_.

.. video:: tutorials/video/destripe.mp4
    :width: 100%
    :caption: Leonardo-DeStripe in Napari
    :align: center

.. video:: tutorials/video/fuse-ill.mp4
    :width: 100%
    :caption: Leonardo-Fuse (along illumination) in Napari
    :align: center

.. video:: tutorials/video/fuse-det.mp4
    :width: 100%
    :caption: Leonardo-Fuse (along detection) in Napari
    :align: center