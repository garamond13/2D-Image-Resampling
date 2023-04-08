# Interpolation-based image scaling
The mpv player user shader for interpolation-based image scaling. Designed for experimental / testing use.

## Usage
- If you place this shader in the same folder as your `mpv.conf`, you can use it with `glsl-shaders-append="~~/interpolation_based_image_scaling.glsl"`.
- The shader is controlled under `user configurable` section by changing macro values and by directly implementing filter kernel.
- Requires `vo=gpu-next`.

## Notes
- The shader is not optimised for speed.
- Antiringing behaves a bit differently compared with the same technique implemented in separated passes.
- In general, you may expect slightly different results from different implementations of resampling algorithms.
- For `linearize` and `delinearize` macros see `pl_shader_linearize` and `pl_shader_delinearize` functions in https://github.com/haasn/libplacebo/blob/master/src/shaders/colorspace.c
- In general for sintax of mpv user shaders see https://libplacebo.org/custom-shaders/
