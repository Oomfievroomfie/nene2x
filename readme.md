# Nene2X

Super small 2x upscaling neural network.

General architecture: simple ReLU convolutional neural network that upscales an image by 2x. Works as a cleanup pass for bilinear: the network gives you 4 values per input pixel, you use them to push around the 4 pixels produced by bilinear.

This is a **single-channel upscaler**. For RGB, my recommendation is to **upscale only the luma channel**, i.e. give it a greyscale input and add the output to every single channel.

The model architecture is designed to be extremely easy to integrate into any existing project that has a post-processing framework capable of handling *any* upscaler. The model runs once per input pixel and produces a set of residuals instead of a single residual.

**Important note**: On linux or nvidia, edit `pyproject.toml` to refer to `torch>=2.4.0` instead of `torch-directml`. I'm on windows+AMD, so I have to use `torch-directml`, and I don't know how to make it pick what or what else based on runtime hardware support.

## I just want to upscale an image

`uv run python infer.py --yonly upscaler.safetensors test/mm_half.png`

Or `--ycgco` to upscale chroma too, or omit the parameter to upscale per RGB channel. `--yonly` is the least prone to color smearing artifacts.

This project uses uv for dependency management; install it first. Don't worry, it's not like conda, it won't bite.

## Examples

TODO

## Pretrained models

The `pretrained/` directory has six pretrained models.

- `nene2x_1pass_3x3_11_11_1` - 290 parameters. Miniature model, trades blows with FSR1. This model is so cheap that it can EASILY run in realtime on a mobile GPU, with negigible cost compared to bilinear alone, upscaling a 720p image to 1440p. Has meaningful artifacts, but see NOTE1. **Recommended for extremely weak hardware (like really old phones/laptops).**
- `nene2x_1pass_3x3_16_16_2` - 772 parameters. Artifacting gone, still realtime, negligible cost, but might be expensive enough to impact framerate on VERY, VERY weak devices (e.g. phones that were low-end in 2016).
- `nene2x_1pass_5x5_28_24_2` - 2124 parameters. Better quality, more expensive. Does a 5x5 convolution, which might be enough to start to have a measurable performance impact (e.g. 1ms or more when upscaling 540p to 1080p) for some of the hardware that you care about.

There are also versions that require an additional 3x3 convolutional pre-pass to turn your greyscale image into a 4-channel feature space; this is still viable with pure image-based post-processing frameworks, but might run into other issues depending on your framework. If your framework is OK with this, then use one of these instead of one of the 1pass versions. (Unless you specifically need the minimum performance of the 1pass 3x3_11_1 version.)

- `nene2x_2pass_3x3_32_16_1` - 956 parameters. **Recommended for general use**.
- `nene2x_2pass_3x3_32_22_2` - 1684 parameters. Recommended for mid-range mobile devices or desktop GPUs in particular.
- `nene2x_2pass_5x5_32_26_3` - 3242 parameters, but SLOW. Even on a strong desktop GPU (RX 6800) it has genuinely meaningful performance cost. Upscaling 1080p to 4k would cost 5~10ms on that RX 6800. **Only recommended for offline (i.e. on-disk) use.**

For 1pass networks, the NxN refers to the size of the first convolution, and the first number (11/16/28) is the number of kernels and outputs of that first pass. Then the second number (11/16/24) is the size of any hidden layers, and the final number (1/2/2) is the number of hidden layers. Hidden layers are fully connected.

For 2pass networks, there is a hidden 3x3 convolutional pre-pass that produces 4 outputs. Then the explicit convolution kernel size is on those outputs instead of the original image. The explicit convolution operates on each of those 4 pre-pass channels one at a time, and the number of outputs from the explicit convolution pass (32/32/32) must be a multiple of 4, and each kernel is only a single channel thick, not 4 channels thick.

Design note for 2 pass networks: I tried letting them be 4 channels thick before, but it resulted in 5x5 convolutions with a meaningful number of output channels (like 14) killing my GPU (200+ms cost when the number of FLOPs happening looked like it should only cost ~5ms) when compiled as GLSL.

The effective feature visibility of a 2pass network is two pixels wider in diameter than the specified kernel size. So 3x3 models can see 5x5 features and 5x5 models can see 7x7 features.

**NOTE1:** See `gridline_remover.omwfx` for an example of a final cleanup pass that can clean up gridline artifacts from turning the sharpness up too far, without blurring the image overall. It improves the quality of this minimum-cost model quite a bit and allows it subjectively outperform FSR1 rather than just trading blows.

## Integration notes

**If your postprocessing framework can handle them, you should use a 2pass model** (assuming you aren't going for absolute minimum cost with the `3x3_11_11_1` model).

For post-processing systems that don't support temporary non-displayed render targets, 1pass models can be made to work by running them once for each output pixel, instead of once for each input pixel. This will make them 4x as expensive, but they'll at least work. Doing this for 2pass models would be impractical, hence providing both 1pass and 2pass models.

## Use as a shader

Run `glsl.py` on one of the networks in `pretrained/` to get a set of GLSL functions that can be ducktaped into any multipass-capable post-processing system (including ones that only allow 1 output texture per pass) to sharpen up a bilinear upscale. See `examples/nn2x.omwfx` as an example of how the ducktaping works. (Note that `omwfx` files don't, at the time of writing, support producing a higher resolution than they take in, so the middle 50% of the input is taken and upscaled, so it's purely for demonstration.)

## Training

Modify `model.py` directly to set the network architecture before training. There's a `gConfig` variable with old configs and my notes on them. Training data is in `train/`, and the images in this repo are the ones I actually used to train the pretrained models. Trains in 10~25 minutes on an RX 6800 depending on network size.

`uv run python train.py train/ --epochs 700`

**NOTE:** Some network geometries are extremely prone to breaking (getting trapped in bad local minima) on unlucky initializations, such that they fail to stop producing some artifacts during training. Either babysit them for the first 100 epochs, checking their output to make sure they aren't going to break, or wrap training in a harness that trains multiple copies of them (at least 4) and picks the best one based on PSNR tests (if you need that harness, build it yourself). Training is done with Leaky ReLUs instead of pure ReLUs to make this problem less common, but it can still happen for extremely small network configurations.

## Results

TODO: automatically take PSNR readings and compare to "waifu2x" (cunet) and "Convolutional upscaling Neural Network, yeah!"

## LLM Usage & Copyright

The architecture of the networks being trained was decided by myself and not the LLM. This readme was written entirely by myself.

The code in this repository is heavily full of LLM boilerplate and has many functions that are entirely LLM-generated. (Unfortunately, this is currently the standard for AI/Machine Learning research.)

All of the training data under `train/` is either public domain (CC0 images from Flickr) or authored by myself.

## License

All code and model data in this repo is licensed under your choice of:

- Creative Commons Zero, any version
- BSD-0
- Unlicense

The images under `train/` are provided for training and may be freely redistributed for that purpose, including heavily transformed versions of them. Any model data produced by training on them belongs to you.
