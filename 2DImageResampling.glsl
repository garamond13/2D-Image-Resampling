//!HOOK MAIN
//!BIND HOOKED
//!WHEN OUTPUT.w OUTPUT.h * MAIN.w MAIN.h * = !
//!WIDTH OUTPUT.w
//!HEIGHT OUTPUT.h
//!DESC 2D Image Resampling

//declarations of kernel filters

//usage example: sinc(x) * sinc(x / RADIUS)
float sinc(float x);

//usage example: sinc(x) * jinc(x / RADIUS * 1.21966989126650445492653885)
float jinc(float x);

//cosine, exponent = 1.0
//hann, exponent = 2.0
//usage example: sinc(x) * power_of_cosine(x / RADIUS, 2.0)
float power_of_cosine(float x, float exponent);

//usage example: sinc(x) * cosine(x / RADIUS)
float cosine(float x);

//tuky, sinc(x) * hann(x / RADIUS * a), 0.0 <= a <= 1.0
//usage example: sinc(x) * hann(x / RADIUS)
float hann(float x);

//usage example: sinc(x) * hamming(x / RADIUS)
float hamming(float x);

//blackman, alpha = 0.16
//usage example: sinc(x) * generalized_blackman(x / RADIUS, 0.16)
float generalized_blackman(float x, float alpha);

//usage example: sinc(x) * blackman(x / RADIUS)
float blackman(float x);

//usage example: sinc(x) * exact_blackman(x / RADIUS)
float exact_blackman(float x);

//usage example: sinc(x) * nuttall(x / RADIUS)
float nuttall(float x);

//usage example: sinc(x) * blackman_nuttall(x / RADIUS)
float blackman_nuttall(float x);

//usage example: sinc(x) * blackman_harris(x / RADIUS)
float blackman_harris(float x);

//usage example: sinc(x) * flat_top(x / RADIUS)
float flat_top(float x);

//usage example: sinc(x) * bohman(x / RADIUS)
float bohman(float x);

//usage example: sinc(x) * kaiser(x / RADIUS, 8.5)
float kaiser(float x, float beta);

//usage example: sinc(x) * parzen(x / RADIUS)
float parzen(float x);

//usage example: sinc(x) * welch(x / RADIUS)
float welch(float x);

//usage example: sinc(x) * gaussian(x / RADIUS, 0.4)
float gaussian(float x, float sigma);

//hann–poisson, sinc(x) * hann(x / RADIUS) * poisson(x / RADIUS, alpha) 
//usage example: sinc(x) * poisson(x / RADIUS, 2.0)
float poisson(float x, float alpha);

//usage example: sinc(x) * cauchy(x / RADIUS, 3.0)
float cauchy(float x, float alpha);

//see https://www.hpl.hp.com/techreports/2007/HPL-2007-179.pdf
//usage example: sinc(x) * said(x, 0.416, 0.61)
float said(float x, float chi, float eta);

//nearest neighbor, RADIUS = 0.5
//usage example: box(x), fixed radius 1.0
float box(float x);

//usage example: nearest_neighbor(x), fixed radius 1.0
float nearest_neighbor(float x);

//usage example: linear(x), fixed radius 1.0
float linear(float x);

//usage example: hermite(x), fixed radius 1.0
float hermite(float x);

//usage example: interpolating_quadratic(x), fixed radius 1.5
float interpolating_quadratic(float x);

//usage example: fsr_kernel(x, 0.4), fixed radius 2.0
float fsr_kernel(float x, float b);

//usage example: bicubic(x, -0.5), fixed radius 2.0
float bicubic(float x, float alpha);

//see https://dl.acm.org/doi/10.1145/378456.378514
//usage example: bc_spline(x, 0.0, 0.5), fixed radius 2.0
float bc_spline(float x, float b, float c);

//usage example: spline16(x), fixed radius 2.0
float spline16(float x);

//usage example: spline36(x), fixed radius 3.0
float spline36(float x);

//usage example: spline64(x), fixed radius 4.0
float spline64(float x);

//usage example: german(x), fixed radius 4.0
float german(float x);

//usage example: magic_kernel_sharp(x), fixed radius 4.5
float magic_kernel_sharp(float x);

//user configurable
//
//the light in which the image will be processed, 1 is linear, 2 is sigmoidal, anything else is expected to be gamma
#define LIGHT 1

//only relevant if LIGHT is set to 2 (sigmoidal)
#define CONTRAST 6.5 //slope of the sigmoid curve
#define MIDPOINT 0.75 //the midpoint of the sigmoid curve

#define RADIUS 3.0 //kernel radius
#define ANTIRINGING 0.0 //reduces ringing, probably should only be used when upsampling (0.0 means off)
#define ANTIALIASING 1.0 //trades between aliasing and ringing, probably should only be used when downsampling (1.0 means off)

float get_weight(float x)
{
    x = abs(x);
    if (x < RADIUS) {
        //kernel filter
        return sinc(x) * hann(x / RADIUS);
    }
    else
        return 0.0;
}
//

//based on https://github.com/ImageMagick/ImageMagick/blob/main/MagickCore/enhance.c
#define sigmoidize(rgba) (MIDPOINT - log(1.0 / ((1.0 / (1.0 + exp(CONTRAST * (MIDPOINT - 1.0))) - 1.0 / (1.0 + exp(CONTRAST * MIDPOINT))) * rgba + 1.0 / (1.0 + exp(CONTRAST * MIDPOINT))) - 1.0) / CONTRAST)
#define desigmoidize(rgba) (1.0 / (1.0 + exp(CONTRAST * (MIDPOINT - rgba))) - 1.0 / (1.0 + exp(CONTRAST * MIDPOINT))) / ( 1.0 / (1.0 + exp(CONTRAST * (MIDPOINT - 1.0))) - 1.0 / (1.0 + exp(CONTRAST * MIDPOINT)))

//main algorithm
vec4 hook()
{
    //HOOKED_pos == texture coordinates [0.0, 1.0], vec2(u, v)
    //input_size == texture dimensions, vec2(width, height)
    //target_size == output dimesions, vec2(width, height)
    //HOOKED_pt == 1.0 / input_size, vec2(1.0 / width, 1.0 / height)
    //HOOKED_raw == texture itself
    //linearize == a macro for linearization, provided by libplacebo 
    //delinearize == a macro for delinearization, provided by libplacebo
    //HOOKED_mul == coefficient to rescale sampled value to [0.0, 1.0], float 
    
    vec2 fcoord = fract(HOOKED_pos * input_size - 0.5);
    vec2 base = HOOKED_pos - fcoord * HOOKED_pt;
    vec4 color;
    vec4 csum = vec4(0.0);
    vec2 weight;
    float wsum = 0.0;
    
    //antiringing
    vec4 low = vec4(1e9);
    vec4 high = vec4(-1e9);

    //only relevant for downsampling, when upsampling scale should be and is set to 1.0
    vec2 scale;
    scale.x = input_size.x / target_size.x < 1.0 ? 1.0 : input_size.x / target_size.x * ANTIALIASING;
    scale.y = input_size.y / target_size.y < 1.0 ? 1.0 : input_size.y / target_size.y * ANTIALIASING;
    
    //determine sampling radiuses
    float ry = ceil(RADIUS * scale.y); //final kernel height = 2 * ry
    float rx = ceil(RADIUS * scale.x); //final kernel width = 2 * rx

    for (float y = 1.0 - ry; y <= ry; y++) {
        weight.y = get_weight((y - fcoord.y) / scale.y);
        for (float x = 1.0 - rx; x <= rx; x++) {
            weight.x = get_weight((x - fcoord.x) / scale.x);
            if (LIGHT == 1)
                color = linearize(textureLod(HOOKED_raw, base + HOOKED_pt * vec2(x, y), 0.0) * HOOKED_mul);
            else if (LIGHT == 2)
                color = sigmoidize(clamp(linearize(textureLod(HOOKED_raw, base + HOOKED_pt * vec2(x, y), 0.0) * HOOKED_mul), 0.0, 1.0));
            else
                color = textureLod(HOOKED_raw, base + HOOKED_pt * vec2(x, y), 0.0) * HOOKED_mul;
            csum += color * weight.x * weight.y;
            wsum += weight.x * weight.y;
            
            //antiringing
            if (ANTIRINGING > 0.0 && x >= 0.0 && y >= 0.0 && x <= 1.0 && y <= 1.0) {
                low = min(low, color);
                high = max(high, color);
            }
        }
    }
    csum /= wsum; //normalize color values
    
    //antiringing
    if (ANTIRINGING > 0.0)
        csum = mix(csum, clamp(csum, low, high), ANTIRINGING);
    
    //mpv and libplacebo should do the clamping as necessary
    if(LIGHT == 1)
        return delinearize(csum);
    else if (LIGHT == 2)
        return delinearize(desigmoidize(csum));
    else
        return csum;
}

//source corecrt_math_defines.h
#define M_PI 3.14159265358979323846 // pi
#define M_PI_2 1.57079632679489661923 // pi/2
#define M_SQRT1_2 0.707106781186547524401 // 1/sqrt(2)

#define EPSILON 1.192093e-7

//declarations of math functions
float bessel_j1(float x);
float bessel_i0(float x);

//definitions of kernel filters

float sinc(float x)
{
    //should be (x == 0), but it can cause unexplainable artifacts
    if (x < EPSILON)
        return 1.0;
    else
        return sin(M_PI * x) / (M_PI * x);
}

float jinc(float x)
{
    //should be (x == 0), but make it consistent with sinc
    if (x < EPSILON)
        return 1.0;
    else
        return 2.0 * bessel_j1(M_PI * x) / (M_PI * x);
}

float power_of_cosine(float x, float exponent)
{
    return pow(cos(M_PI_2 * x), exponent);
}

float cosine(float x)
{
    return cos(M_PI_2 * x);
}

float hann(float x)
{
    return 0.5 + 0.5 * cos(M_PI * x);
}

float hamming(float x)
{
    return 0.54 + 0.46 * cos(M_PI * x);
}

float generalized_blackman(float x, float alpha)
{
    return (1.0 - alpha) / 2.0 + 0.5 * cos(M_PI * x) + alpha / 2.0 * cos(2.0 * M_PI * x);
}

float blackman(float x)
{
    return 0.42 + 0.5 * cos(M_PI * x) + 0.08 * cos(2.0 * M_PI * x);
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

float kaiser(float x, float beta)
{
    //beta = pi * alpha
    return bessel_i0(beta * sqrt(1.0 - x * x)) / bessel_i0(beta);
}

float parzen(float x)
{
    if (x < 0.5)
        return 1.0 - 6.0 * x * x * (1.0 - x);
    else //x < 1.0
        return 2.0 * (1.0 - x) * (1.0 - x) * (1.0 - x);
}

float welch(float x)
{
    return 1.0 - x * x;
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
    if (x < 0.5)
        return 1.0;
    else
        return 0.0;
}

float linear(float x)
{
    if(x < 1.0)
        return 1.0 - x;
    else
        return 0.0;
}

float hermite(float x)
{
    if(x < 1.0)
        return (2.0 * x - 3.0) * x * x + 1.0;
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

//source https://github.com/GPUOpen-Effects/FidelityFX-FSR
float fsr_kernel(float x, float b)
{
    if (x < 2.0)
        return (1.0 / (2.0 * b - b * b) * (b * x * x - 1.0) * (b * x * x - 1.0) - (1.0 / (2.0 * b - b * b) - 1.0)) * (0.25 * x * x - 1.0) * (0.25 * x * x - 1.0);
    else
        return 0.0;
}

float bicubic(float x, float alpha)
{
    if (x < 1.0)
        return (alpha + 2.0) * x * x * x - (alpha + 3.0) * x * x + 1.0;
    else if (x < 2.0)
        return alpha * x * x * x - 5.0 * alpha * x * x + 8.0 * alpha * x - 4.0 * alpha;
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

//definitions of math functions

//bessel function of the first kind (J1), based on https://github.com/ImageMagick/ImageMagick/blob/main/MagickCore/resize.c
//
float j1(float x)
{
    float p, q;
    const float pone[] = {
        0.581199354001606143928050809e+21,
        -0.6672106568924916298020941484e+20,
        0.2316433580634002297931815435e+19,
        -0.3588817569910106050743641413e+17,
        0.2908795263834775409737601689e+15,
        -0.1322983480332126453125473247e+13,
        0.3413234182301700539091292655e+10,
        -0.4695753530642995859767162166e+7,
        0.270112271089232341485679099e+4
    };
    const float qone[] = {
        0.11623987080032122878585294e+22,
        0.1185770712190320999837113348e+20,
        0.6092061398917521746105196863e+17,
        0.2081661221307607351240184229e+15,
        0.5243710262167649715406728642e+12,
        0.1013863514358673989967045588e+10,
        0.1501793594998585505921097578e+7,
        0.1606931573481487801970916749e+4,
        0.1e+1
    };
    p = pone[8];
    q = qone[8];
    for (int i = 7; i >= 0; --i) {
        p = p * x * x + pone[i];
        q = q * x * x + qone[i];
    }
    return p / q;
}

float p1(float x)
{
    float p, q;
    const float pone[] = {
        0.352246649133679798341724373e+5,
        0.62758845247161281269005675e+5,
        0.313539631109159574238669888e+5,
        0.49854832060594338434500455e+4,
        0.2111529182853962382105718e+3,
        0.12571716929145341558495e+1
    };
    const float qone[] = {
        0.352246649133679798068390431e+5,
        0.626943469593560511888833731e+5,
        0.312404063819041039923015703e+5,
        0.4930396490181088979386097e+4,
        0.2030775189134759322293574e+3,
        0.1e+1
    };
    p = pone[5];
    q = qone[5];
    for (int i = 4; i >= 0; --i) {
        p = p * (8.0 / x) * (8.0 / x) + pone[i];
        q = q * (8.0 / x) * (8.0 / x) + qone[i];
    }
    return p / q;
}

float q1(float x)
{
    float p, q;
    const float pone[] = {
        0.3511751914303552822533318e+3,
        0.7210391804904475039280863e+3,
        0.4259873011654442389886993e+3,
        0.831898957673850827325226e+2,
        0.45681716295512267064405e+1,
        0.3532840052740123642735e-1
    };
    const float qone[] = {
        0.74917374171809127714519505e+4,
        0.154141773392650970499848051e+5,
        0.91522317015169922705904727e+4,
        0.18111867005523513506724158e+4,
        0.1038187585462133728776636e+3,
        0.1e+1
    };
    p = pone[5];
    q = qone[5];
    for (int i = 4; i >= 0; --i) {
        p = p * (8.0 / x) * (8.0 / x) + pone[i];
        q = q * (8.0 / x) * (8.0 / x) + qone[i];
    }
    return p / q;
}

float bessel_j1(float x)
{
    float p, q;
    if (x == 0.0)
        return 0.0;
    p = x;
    if (x < 0.0)
        x = -x;
    if (x < 8.0)
        return(p * j1(x));
    q = sqrt(2.0 / (M_PI * x)) * (p1(x) * (M_SQRT1_2 * (sin(x) - cos(x))) - 8.0 / x * q1(x) * (-M_SQRT1_2 * (sin(x) + cos(x))));
    if (p < 0.0)
        q = -q;
    return q;
}
//

//modified bessel function of the first kind (I0), based on https://github.com/ImageMagick/ImageMagick/blob/main/MagickCore/resize.c
float bessel_i0(float x)
{
  float sum = 1.0;
  float y = x * x / 4.0;
  float t = y;
  for (float i = 2.0; t > EPSILON; ++i) {
    sum += t;
    t *= y / (i * i);
  }
  return sum;
}
