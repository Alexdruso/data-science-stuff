# Implicit Neural Representations: Fitting Images with Coordinate Networks

A small, self-contained example of **implicit neural representations (INRs)**: instead of storing an
image as a grid of pixels, train a neural network `f_θ(x, y) → (R, G, B)` that maps *continuous* pixel
coordinates to colours. The image becomes a function, and the network's weights become the image.

The notebook tells the story in three acts:

1. **A plain ReLU MLP fails.** Trained on raw `(x, y)` coordinates it only recovers a blurry,
   low-frequency version of the image — the *spectral bias* of neural networks
   (Rahaman et al., 2019).
2. **Two classic fixes.** Random **Fourier features** (Tancik et al., 2020) lift the coordinates to
   `[sin(2πBx), cos(2πBx)]` before the MLP; **SIREN** (Sitzmann et al., 2020) replaces ReLU with
   sine activations and a principled initialisation. Both fit the image sharply.
3. **Application: arbitrary-scale upscaling.** Because the trained network is a continuous function,
   it can be sampled on *any* grid — including resolutions it never saw and non-integer scale
   factors. We train on a low-resolution image and render it at 4× (and 2.5×), comparing against
   bicubic interpolation and the ground truth.

An honest caveat, spelled out in the notebook: an INR fitted to a *single* low-resolution image
interpolates smoothly but cannot invent detail that isn't in its training signal, so it lands close
to bicubic. Learned priors across many images — e.g. **LIIF** (Chen et al., 2021) — are what push
implicit super-resolution past classical interpolation. This example demonstrates the *mechanism*
(one compact network, decoded at any resolution), which is the same machinery those methods build on.

## Layout

```
├── data/lighthouse.png               # 256×384 sample image (Kodak dataset, kodim19)
├── notebooks/
│   └── implicit_image_fitting.ipynb  # the narrative — start here
├── outputs/                          # generated figures (gitignored)
└── src/
    ├── image.py                      # PNG loading, area downsampling, PSNR
    ├── models.py                     # ReluMLP, FourierFeatureMLP, Siren
    └── training.py                   # coordinate grid, full-batch fit loop, render()
```

## Running it

From the repo root (see the root README for environment setup):

```bash
source .venv/bin/activate
jupyter notebook implicit_neural_representations/notebooks/implicit_image_fitting.ipynb
```

Everything runs on CPU in a few minutes; a GPU is used automatically if available. No dependencies
beyond the repo's core ones (`torch`, `numpy`, `matplotlib`).

The sample image is `kodim19` ("lighthouse") from the [Kodak Lossless True Color Image
Suite](https://r0k.us/graphics/kodak/), released by Kodak for unrestricted use, downsampled to keep
the repo small.

## References

- Sitzmann, V., Martel, J. N. P., Bergman, A. W., Lindell, D. B., Wetzstein, G. (2020).
  *Implicit Neural Representations with Periodic Activation Functions* (SIREN). NeurIPS 2020.
  [arXiv:2006.09661](https://arxiv.org/abs/2006.09661)
- Tancik, M., Srinivasan, P. P., Mildenhall, B., Fridovich-Keil, S., Raghavan, N., Singhal, U.,
  Ramamoorthi, R., Barron, J. T., Ng, R. (2020). *Fourier Features Let Networks Learn High
  Frequency Functions in Low Dimensional Domains*. NeurIPS 2020.
  [arXiv:2006.10739](https://arxiv.org/abs/2006.10739)
- Rahaman, N., Baratin, A., Arpit, D., Draxler, F., Lin, M., Hamprecht, F. A., Bengio, Y.,
  Courville, A. (2019). *On the Spectral Bias of Neural Networks*. ICML 2019.
  [arXiv:1806.08734](https://arxiv.org/abs/1806.08734)
- Mildenhall, B., Srinivasan, P. P., Tancik, M., Barron, J. T., Ramamoorthi, R., Ng, R. (2020).
  *NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis*. ECCV 2020.
  [arXiv:2003.08934](https://arxiv.org/abs/2003.08934)
- Chen, Y., Liu, S., Wang, X. (2021). *Learning Continuous Image Representation with Local
  Implicit Image Function* (LIIF). CVPR 2021.
  [arXiv:2012.09161](https://arxiv.org/abs/2012.09161)
- Park, J. J., Florence, P., Straub, J., Newcombe, R., Lovegrove, S. (2019). *DeepSDF: Learning
  Continuous Signed Distance Functions for Shape Representation*. CVPR 2019.
  [arXiv:1901.05103](https://arxiv.org/abs/1901.05103)
- Mescheder, L., Oechsle, M., Niemeyer, M., Nowozin, S., Geiger, A. (2019). *Occupancy Networks:
  Learning 3D Reconstruction in Function Space*. CVPR 2019.
  [arXiv:1812.03828](https://arxiv.org/abs/1812.03828)
- Stanley, K. O. (2007). *Compositional Pattern Producing Networks: A Novel Abstraction of
  Development*. Genetic Programming and Evolvable Machines, 8(2). (Early coordinate-based networks.)
