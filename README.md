# Image Processors Module

A Python library of composable image "processor" classes for building deep-learning data pipelines
from medical images. Processors operate on dictionaries of features (e.g. `image`, `annotation`)
and handle preprocessing and augmentation — resampling, normalization, bounding-box cropping,
one-hot encoding, noise, connected-component cleanup — on the way into TensorFlow records/datasets
or PyTorch datasets. Originally built to feed the companion `Data_Generators` and
`Make_Single_Images` repos.

```
pip install ImageProcessorsModule
```

## Key components (`src/Processors`)

- `MakeTFRecordProcessors.py` — SimpleITK/NumPy processors that turn images and annotations into feature dictionaries for record writing
- `TFRecordWriter.py` — multithreaded TFRecord writer
- `TFDataSets/` (`ConstantProcessors`, `RelativeProcessors`, `SpecialProcessors`, `TFGenerator`) — processors applied inside `tf.data` pipelines
- `PyTorchDataSets/` (`ConstantProcessors`, `RelativeProcessors`) — equivalent processors for PyTorch datasets
- `KerasGeneratorProcessors.py` — processors for Keras generators

`Fill_In_Segments_sitk.py` (at the repo root) provides standalone SimpleITK/scikit-image
utilities for cleaning segmentation masks (hole filling, largest-connected-component filtering).

## Requirements

Python >= 3.8. Depends on `numpy`, `matplotlib`, `SimpleITK`, `scikit-image`, `opencv-python`,
and the author's `PlotScrollNumpyArrays` and `NiftiResampler` packages. TensorFlow and PyTorch
are required by their respective submodules but are not declared as package dependencies —
install the framework you use.

## License

GNU Affero General Public License v3.
