# 2D Image Resampling
2D Image Resampling is a general single pass resampling algorithm made for experimental / testing use. It's designed to work as mpv player user shader.

## Usage
- If you place this shader in the same folder as your `mpv.conf`, you can use it with `glsl-shaders-append="~~/2DImageResampling.glsl"`.
- The shader is controlled under `user configurable` section by changing macro values and by directly implementing filter kernel.
- Requires `vo=gpu-next`.

## Notes
- The shader is not optimised for speed.
- Antiringing behaves a bit differently compared with the same technique implemented in separated passes.
- In general, you may expect slightly different results from different implementations of resampling algorithms.
