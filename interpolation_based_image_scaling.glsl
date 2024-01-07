//!HOOK MAIN
//!BIND HOOKED
//!WHEN OUTPUT.w OUTPUT.h * MAIN.w MAIN.h * = !
//!WIDTH OUTPUT.w
//!HEIGHT OUTPUT.h
//!DESC interpolation_based_image_scaling

// Forward declarations of kernel filter functions.
//

// Usage example as window: sinc(x / RADIUS)
// Usage example as base: sinc(x / BLUR)
float sinc(float x);

// Usage example as window: jinc(x / RADIUS * 1.2196699 /* first zero */ )
// Usage example as base: jinc(x / BLUR)
float jinc(float x);

// Usage example as window: sphinx(x / RADIUS * 1.4302967 /* first zero */ )
float sphinx(float x);

// Usage example as window: cosine(x / RADIUS)
float cosine(float x);

// cosine: n = 1.
// hann: n = 2.
// box: n = 0.
// Has to be satisfied: n >= 0.
// Usage example as window: power_of_cosine(x / RADIUS, 1.0)
float power_of_cosine(float x, float n);

// Usage example as window: hann(x / RADIUS)
float hann(float x);

// common tukey: a = 0.5.
// hann: a = 1.
// box: a = 0.
// Usage example as window: tukey(x / RADIUS, 0.5)
float tukey(float x, float a);

// Usage example as window: hamming(x / RADIUS)
float hamming(float x);

// common blackman: a = 0.16.
// hann: a = 0.
// Usage example as window: blackman(x / RADIUS, 0.16)
float blackman(float x, float a);

// blackman: n = 1
// common blackman: a = 0.16, n = 1.
// hann: a = 0, n = 1.
// cosine: a = 0, n = 0.5
// box: n = 0.
// Has to be satisfied: n >= 0.
// Has to be satisfied: if n != 1, a <= 0.16.
// Usage example as window: power_of_blackman(x / RADIUS, 0.16, 1.0)
float power_of_blackman(float x, float a, float n);

// Usage example as window: exact_blackman(x / RADIUS)
float exact_blackman(float x);

// Usage example as window: nuttall(x / RADIUS)
float nuttall(float x);

// Usage example as window: blackman_nuttall(x / RADIUS)
float blackman_nuttall(float x);

// Usage example as window: blackman_harris(x / RADIUS)
float blackman_harris(float x);

// Usage example as window: flat_top(x / RADIUS)
float flat_top(float x);

// Usage example as window: bohman(x / RADIUS)
float bohman(float x);

// Usage example as window: welch(x / RADIUS)
float welch(float x);

// Usage example as window: kaiser(x / RADIUS, 8.5)
float kaiser(float x, float beta);

// kaiser: n = 2.
// Has to be satisfied: n >= 0.
// Usage example as window: kaiser_garamond(x / RADIUS, 8.5, 2.0)
float kaiser_garamond(float x, float beta, float n);

// Usage example as window: parzen(x / RADIUS)
float parzen(float x);

// Usage example as window: gaussian(x / RADIUS, 0.4)
float gaussian(float x, float sigma);

// hann–poisson: hann(x / RADIUS) * poisson(x, alpha).
// Usage example as window: poisson(x / RADIUS, 2.0)
float poisson(float x, float alpha);

// Usage example as window: cauchy(x / RADIUS, 3.0)
float cauchy(float x, float alpha);

// linear: n = 1 m = 1.
// welch: n = 2, m = 1.
// box: n -> inf, m <= 1.
// box: m = 0.
// garamond window: m = 1.
// Has to be satisfied: n >= 0, m >= 0.
// Usage example as window: power_of_garamond_window(x / RADIUS, 2.0, 1.0)
float power_of_garamond_window(float x, float n, float m);

// gaussian: n = 2.
// box: n -> inf.
// Has to be satisfied: s != 0, n >= 0.
// Usage example as window: generalized_normal_window(x, 2.0, 3.0)
float generalized_normal_window(float x, float s, float n);

// Has to be satisfied: eta != 2.
// Usage example as window: said(x, 0.416, 0.61)
float said(float x, float chi, float eta);

// Usage example: box(x)
float box(float x);

// Usage example: nearest_neighbor(x), fixed radius 1.0
float nearest_neighbor(float x);

// Usage example: linear(x), fixed radius 1.0
float linear(float x);

// Usage example: interpolating_quadratic(x), fixed radius 1.5
float interpolating_quadratic(float x);

// Has to be satisfied: b != 0, b != 2.
// Usage example: fsr_kernel(x, 0.4), fixed radius 2.0
float fsr_kernel(float x, float b);

// original fsr kernel: c = 1.
// Has to be satisfied: b != 0, b != 2, c != 0.
// Usage example: modified_fsr_kernel(x, 0.4, 1.0), fixed radius 2.0
float modified_fsr_kernel(float x, float b, float c);

// Usage example: bicubic(x, -0.5), fixed radius 2.0
float bicubic(float x, float a);

// hermite: b = 0, c = 0.
// spline: b = 1, c = 0.
// catmull_rom: b = 0, c = 0.5.
// mitchell: b = 1 / 3, c = 1 / 3.
// robidoux: b = 12 / (19 + 9 * sqrt(2)) ≈ 0.37821575509399866, c = 113 / (58 + 216 * sqrt(2)) ≈ 0.31089212245300067.
// robidouxsharp: b = 6 / (13 + 7 * sqrt(2)) ≈ 0.2620145123990142, c = 7 / (2 + 12 * sqrt(2)) ≈ 0.3689927438004929.
// robidouxsoft: b = (9 - 3 * sqrt(2)) / 7 ≈ 0.67962275898295921, c ≈ 0.1601886205085204.
// Usage example: bc_spline(x, 0.0, 0.5), fixed radius 2.0.
float bc_spline(float x, float b, float c);

// Usage example: spline16(x), fixed radius 2.0
float spline16(float x);

// Usage example: spline36(x), fixed radius 3.0
float spline36(float x);

// Usage example: spline64(x), fixed radius 4.0
float spline64(float x);

// Usage example: german(x), fixed radius 4.0
float german(float x);

// Usage example: magic_kernel_sharp(x), fixed radius 4.5
float magic_kernel_sharp(float x);

//

// User configurable
//

// LIGHTS LIST (not configurable, use as it is)
#define LIGHT_LINEAR 1
#define LIGHT_SIGMOIDAL 2
#define LIGHT_GAMMA 3

// The light in which the image will be processed.
// See LIGHTS LIST above for available options.
#define LIGHT LIGHT_GAMMA

// Only relevant if LIGHT is set to LIGHT_SIGMOIDAL.
#define CONTRAST 6.0 // Slope of the sigmoid curve. Has to be > 0.
#define MIDPOINT 0.6 // Midpoint of the sigmoid curve. Has to be in range [0, 1].

// Use cylindrical resampling.
#define USE_CYLINDRICAL false // Boolean, false or true.

#define RADIUS 3.0 // Kernel radius. Has to be > 0.
#define BLUR 1.0 // Kernel blur. Blures or sharpens the kernel, 1.0 is no effect.
#define ANTIRINGING 0.0 // Probably should only be used when upsampling. 0.0 means off. Has to be in range [0, 1].

// Not configurable, except for see below.
float get_weight(float x)
{
    if (x < RADIUS) {
        
        // This is configurable.
        // Set your kernel function here as return value.
        return sinc(x / BLUR) * hann(x / RADIUS);
    
    }
    else
        return 0.0;
}

//

// Based on https://github.com/ImageMagick/ImageMagick/blob/main/MagickCore/enhance.c
#define C CONTRAST
#define M MIDPOINT
#define sigmoidize(rgba) (M - log(1.0 / ((1.0 / (1.0 + exp(C * (M - 1.0))) - 1.0 / (1.0 + exp(C * M))) * (rgba) + 1.0 / (1.0 + exp(C * M))) - 1.0) / C)
#define desigmoidize(rgba) (1.0 / (1.0 + exp(C * (M - (rgba)))) - 1.0 / (1.0 + exp(C * M))) / ( 1.0 / (1.0 + exp(C * (M - 1.0))) - 1.0 / (1.0 + exp(C * M)))

// The main algorithm (main function).
vec4 hook()
{
    // HOOKED_pos: Normalized texture coordinates vec2(u, v).
    // input_size: Texture dimensions vec2(width, height).
    // target_size: Output dimesions vec2(width, height).
    // HOOKED_pt: Texel size vec2(1.0 / width, 1.0 / height).
    // HOOKED_raw: Texture itself.
    // linearize: A macro for linearization, provided by libplacebo.
    // delinearize: A macro for delinearization, provided by libplacebo.
    // HOOKED_mul: Coefficient to rescale sampled value to [0.0, 1.0]. float.
    
    const vec2 fcoord = fract(HOOKED_pos * input_size - 0.5);
    const vec2 base = HOOKED_pos - fcoord * HOOKED_pt;
    vec4 color;
    vec4 csum = vec4(0.0);
    vec2 weight;
    float wsum = 0.0;
    
    // Antiringing.
    vec4 lo = vec4(1e9);
    vec4 hi = vec4(-1e9);

    // When downsampling we need an actual scale.
    // When upsampling scale has to be set to 1 (ignored).
    // We only expect proportional scaling!
    const float scale = min(target_size.x / input_size.x, 1.0);

    const int bound = int(ceil(RADIUS / scale));
    for (int y = 1 - bound; y <= bound; y++) {
        
        // For the cylindrical resampling simply ignore the y component by setting it to 1.
        if (USE_CYLINDRICAL)
            weight.y = 1.0;
        else
            weight.y = get_weight(abs((float(y) - fcoord.y) * scale));
        
        for (int x = 1 - bound; x <= bound; x++) {
            if (USE_CYLINDRICAL)
                weight.x = get_weight(length(vec2(x, y) - fcoord) * scale);
            else
                weight.x = get_weight(abs((float(x) - fcoord.x) * scale));
            
            // Sample color.
            if (LIGHT == LIGHT_LINEAR)
                color = linearize(textureLod(HOOKED_raw, base + HOOKED_pt * vec2(x, y), 0.0) * HOOKED_mul);
            else if (LIGHT == LIGHT_SIGMOIDAL)
                color = sigmoidize(clamp(linearize(textureLod(HOOKED_raw, base + HOOKED_pt * vec2(x, y), 0.0) * HOOKED_mul), 0.0, 1.0));
            else // LIGHT_GAMMA
                color = textureLod(HOOKED_raw, base + HOOKED_pt * vec2(x, y), 0.0) * HOOKED_mul;
            
            csum += color * weight.x * weight.y;
            wsum += weight.x * weight.y;
            
            // Antiringing.
            if (ANTIRINGING > 0.0 && y >= 0 && y <= 1 && x >= 0 && x <= 1) {
                lo = min(lo, color);
                hi = max(hi, color);
            }
        }
    }
    csum /= wsum; // Normalize color values.
    
    // Antiringing.
    if (ANTIRINGING > 0.0)
        csum = mix(csum, clamp(csum, lo, hi), ANTIRINGING);
    
    // MPV and libplacebo should do the clamping as necessary.
    if(LIGHT == LIGHT_LINEAR)
        return delinearize(csum);
    else if (LIGHT == LIGHT_SIGMOIDAL)
        return delinearize(desigmoidize(csum));
    else // LIGHT_GAMMA
        return csum;
}

// Source corecrt_math_defines.h.
// Rounded to 7 decimal places.
#define M_PI 3.1415927 // pi
#define M_PI_2 1.5707963 // pi/2
#define M_PI_4 0.7853982 // pi/4
#define M_2_PI 0.6366198 // 2/pi
#define M_SQRT2 1.4142136 // sqrt(2)

#define EPSILON 1e-6

// Forward declarations of math functions.
float bessel_J1(float x);
float bessel_I0(float x);
float bessel_j1(float x);

// Definitions of kernel filter functions.
//

float sinc(float x)
{
    // Should be (x == 0).
    if (x < EPSILON)
        return 1.0;
    else
        return sin(M_PI * x) / (M_PI * x);
}

float jinc(float x)
{
    // Should be (x == 0).
    if (x < EPSILON)
        return 1.0;
    else
        return 2.0 * bessel_J1(M_PI * x) / (M_PI * x);
}

// Source https://github.com/haasn/mp/commit/7d10c9b76f39bfd2fe606b8702b39888d117c685
float sphinx(float x)
{
    // Should be (x == 0).
    if (x < EPSILON)
        return 1.0;
    else
        return 3.0 * bessel_j1(M_PI * x) / (M_PI * x);
}

float cosine(float x)
{
    return cos(M_PI_2 * x);
}

float power_of_cosine(float x, float n)
{
    return pow(cos(M_PI_2 * x), n);
}

float hann(float x)
{
    return 0.5 + 0.5 * cos(M_PI * x);
}

float tukey(float x, float a)
{
    return 0.5 + 0.5 * cos(M_PI * x * a);
}

float hamming(float x)
{
    return 0.54 + 0.46 * cos(M_PI * x);
}

float blackman(float x, float a)
{
    return (1.0 - a) / 2.0 + 0.5 * cos(M_PI * x) + a / 2.0 * cos(2.0 * M_PI * x);
}

float power_of_blackman(float x, float a, float n)
{
    return pow((1.0 - a) / 2.0 + 0.5 * cos(M_PI * x) + a / 2.0 * cos(2.0 * M_PI * x), n);
}

float exact_blackman(float x)
{
    return 7938.0 / 18608.0 + 9240.0 / 18608.0 * cos(M_PI * x) + 1430.0 / 18608.0 * cos(2.0 * M_PI * x);
}

float nuttall(float x)
{
    return 0.355768 + 0.487396 * cos(M_PI * x) + 0.144232 * cos(2.0 * M_PI * x) + 0.012604 * cos(3.0 * M_PI * x);
}

float blackman_nuttall(float x)
{
    return 0.3635819 + 0.4891775 * cos(M_PI * x) + 0.1365995 * cos(2.0 * M_PI * x) + 0.0106411 * cos(3.0 * M_PI * x);
}

float blackman_harris(float x)
{
    return 0.35875 + 0.48829 * cos(M_PI * x) + 0.14128 * cos(2.0 * M_PI * x) + 0.01168 * cos(3.0 * M_PI * x);
}

float flat_top(float x)
{
    return 0.21557895 + 0.41663158 * cos(M_PI * x) + 0.277263158 * cos(2.0 * M_PI * x) + 0.083578947 * cos(3.0 * M_PI * x) + 0.006947368 * cos(4.0 * M_PI * x);
}

float bohman(float x)
{
    return (1.0 - x) * cos(M_PI *  x) + sin(M_PI * x) / M_PI;
}

float welch(float x)
{
    return 1.0 - x * x;
}

float kaiser(float x, float beta)
{
    return bessel_I0(beta * sqrt(1.0 - x * x)) / bessel_I0(beta);
}

float kaiser_garamond(float x, float beta, float n)
{
    return bessel_I0(beta * sqrt(1.0 - pow(x, n))) / bessel_I0(beta);
}

float parzen(float x)
{
    if (x < 0.5)
        return 1.0 - 6.0 * x * x * (1.0 - x);
    else //x < 1.0
        return 2.0 * (1.0 - x) * (1.0 - x) * (1.0 - x);
}

float gaussian(float x, float sigma)
{
    return exp(-(x * x / (2.0 * sigma * sigma)));
}

float poisson(float x, float alpha)
{
    return exp(-alpha * x);
}

float cauchy(float x, float alpha)
{
    return 1.0 / (1.0 + alpha * alpha * x * x);
}

float power_of_garamond_window(float x, float n, float m)
{
    // Should be (n == 0).
    if (n < EPSILON)
        return 1.0;
    else
        return pow(1.0 - pow(x, n), m);
}

float generalized_normal_window(float x, float s, float n)
{
    return exp(-pow(x / s, n));
}

float said(float x, float chi, float eta)
{
    return cosh(sqrt(2.0 * eta) * M_PI * chi * x / (2.0 - eta)) * exp(-((M_PI * chi * x / (2.0 - eta)) * (M_PI * chi * x / (2.0 - eta))));
}

float box(float x)
{
    return 1.0;
}

float nearest_neighbor(float x)
{
    if (USE_CYLINDRICAL) {
        if (x <= 0.5 * M_SQRT2)
            return 1.0;
        else
            return 0.0;
    }
    else {
        if (x <= 0.5)
            return 1.0;
        else
            return 0.0;
    }
}

float linear(float x)
{
    if(x < 1.0)
        return 1.0 - x;
    else
        return 0.0;
}

float interpolating_quadratic(float x)
{
    if (x < 0.5)
        return -2.0 * x * x + 1.0;
    else if (x < 1.5)
        return x * x - 2.5 * x + 1.5;
    else
        return 0.0;
}

// Source https://github.com/GPUOpen-Effects/FidelityFX-FSR
float fsr_kernel(float x, float b)
{
    if (x < 2.0)
        return (1.0 / (2.0 * b - b * b) * (b * x * x - 1.0) * (b * x * x - 1.0) - (1.0 / (2.0 * b - b * b) - 1.0)) * (0.25 * x * x - 1.0) * (0.25 * x * x - 1.0);
    else
        return 0.0;
}

// Based on https://github.com/GPUOpen-Effects/FidelityFX-FSR
float modified_fsr_kernel(float x, float b, float c)
{
    if (x < 2.0)
        return (1.0 / (2.0 * b - b * b) * (b / (c * c) * x * x - 1.0) * (b / (c * c) * x * x - 1.0) - (1.0 / (2.0 * b - b * b) - 1.0)) * (0.25 * x * x - 1.0) * (0.25 * x * x - 1.0);
    else
        return 0.0;
}

float bicubic(float x, float a)
{
    if (x < 1.0)
        return (a + 2.0) * x * x * x - (a + 3.0) * x * x + 1.0;
    else if (x < 2.0)
        return a * x * x * x - 5.0 * a * x * x + 8.0 * a * x - 4.0 * a;
    else
        return 0.0;
}

float bc_spline(float x, float b, float c)
{
    if (x < 1.0)
        return ((12.0 - 9.0 * b - 6.0 * c) * x * x * x + (-18.0 + 12.0 * b + 6.0 * c) * x * x + (6.0 - 2.0 * b)) / 6.0;
    else if (x < 2.0)
        return ((-b - 6.0 * c) * x * x * x + (6.0 * b + 30.0 * c) * x * x + (-12.0 * b - 48.0 * c) * x + (8.0 * b + 24.0 * c)) / 6.0;
    else
        return 0.0;
}

float spline16(float x)
{
    if (x < 1.0)
        return ((x - 9.0 / 5.0) * x - 1.0 / 5.0 ) * x + 1.0;
    else if (x < 2.0)
        return ((-1.0 / 3.0 * (x - 1.0) + 4.0 / 5.0) * (x - 1.0) - 7.0 / 15.0) * (x - 1.0);
    else
        return 0.0;
}

float spline36(float x)
{
    if (x < 1.0)
        return ((13.0 / 11.0 * x - 453.0 / 209.0) * x - 3.0 / 209.0) * x + 1.0;
    else if (x < 2.0)
        return ((-6.0 / 11.0 * (x - 1.0) + 270.0 / 209.0) * (x - 1.0) - 156.0 / 209.0) * (x - 1.0);
    else if (x < 3.0)
        return ((1.0 / 11.0 * (x - 2.0) - 45.0 / 209.0) * (x - 2.0) +  26.0 / 209.0) * (x - 2.0);
    else
        return 0.0;
}

float spline64(float x)
{
    if (x < 1.0)
        return ((49.0 / 41.0 * x - 6387.0 / 2911.0) * x - 3.0 / 2911.0) * x + 1.0;
    else if (x < 2.0)
        return ((-24.0 / 41.0 * (x - 1.0) + 4032.0 / 2911.0) * (x - 1.0) - 2328.0 / 2911.0) * (x - 1.0);
    else if (x < 3.0)
        return ((6.0 / 41.0 * (x - 2.0) - 1008.0 / 2911.0) * (x - 2.0) + 582.0 / 2911.0) * (x - 2.0);
    else if (x < 4.0)
        return ((-1.0 / 41.0 * (x - 3.0) + 168.0 / 2911.0) * (x - 3.0) - 97.0 / 2911.0) * (x - 3.0);
    else
        return 0.0;
}

float german(float x)
{
    if (x < 1.0)
        return 1.0 / 24.0 * x * x * x * x + 185.0 / 144.0 * x * x * x - 335.0 / 144.0 * x * x + 1.0;
    else if ( x < 2.0)
        return -1.0 / 16.0 * x * x * x * x - 29.0 / 144.0 * x * x * x + 17.0 / 8.0 * x * x - 145.0 / 36.0 * x + 13.0 / 6.0;
    else if ( x < 3.0)
        return 1.0 / 48.0 * x * x * x * x - 7.0 / 48.0 * x * x * x + 5.0 / 18.0 * x * x + 1.0 / 36.0 * x - 1.0 / 3.0;
    else if (x < 4.0)
        return 1.0 / 144.0 * x * x * x - 11.0 / 144.0 * x * x + 5.0 / 18.0 * x - 1.0 / 3.0;
    else
        return 0.0;
}

float magic_kernel_sharp(float x)
{
    if (x < 0.5)
        return 577.0 / 576.0 - 239.0 / 144.0 * x * x;
    else if (x < 1.5)
        return 1.0 / 144.0 * (140.0 * x * x - 379.0 * x + 239.0);
    else if (x < 2.5)
        return -1.0 / 144.0 * (24.0 * x * x - 113.0 * x + 130.0);
    else if (x < 3.5)
        return 1.0 / 144.0 * (4.0 * x * x - 27.0 * x + 45.0);
    else if (x < 4.5)
        return -1.0 / 1152.0 * (4.0 * x * x - 36.0 * x + 81.0);
    else
        return 0.0;
}

//

// Definitions of math functions
//

// Bessel function of the first kind, order one (J1).
// Aproximation optimized for use in shaders.
float bessel_J1(float x)
{
    if (x < 2.2931157)
        return x / 2.0 - x * x * x / 16.0 + x * x * x * x * x / 384.0 - x * x * x * x * x * x * x / 18432.0;
    else
        sqrt(M_2_PI / x) * (1.0 + 0.1875 / (x * x) - 0.1933594 / (x * x * x * x)) * cos(x - 3.0 * M_PI_4 + 0.375 / x - 0.1640625 / (x * x * x));
}

// Modified Bessel function of the first kind, order zero (I0).
// Aproximation optimized for use in shaders.
float bessel_I0(float x)
{
    if (x < 4.9706658)
        return 1.0 + x * x / 4.0 + x * x * x * x / 64.0 + x * x * x * x * x * x / 2304.0 + x * x * x * x * x * x * x * x / 147456.0;
    else
        inversesqrt(2.0 * M_PI * (x)) * exp(x);
}

// Spherical bessel function of the first kind, order one (j1).
float bessel_j1(float x)
{
    return sin(x) / (x * x) - cos(x) / x;
}