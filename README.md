# 2D Image Resampling
2D Image Resampling is a general resampling algorithm made for experimental / testing use. It's designed to work as mpv player shader.

## Usage
If you place this shader in the same folder as your `mpv.conf`, you can use it with `glsl-shaders-append="~~/2DImageResampling.glsl"`.
The shader is controlled under `user configurable` section by changing macro values and by directly implementing filter kernel.

## Notes
- the shader is not optimised for speed
- antiringing behaves a bit differently compared with the same technique implemented in separated passes
