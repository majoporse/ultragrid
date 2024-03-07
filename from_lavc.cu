#include "from_lavc.h"
#include <memory>
#include <vector>
#include <iostream>
#include <fstream>
#include <iomanip>
#include "../src/color.h"
#include "conv_utils.h"
#include <map>

#define R R_SHIFT_IDX
#define G G_SHIFT_IDX
#define B B_SHIFT_IDX

__device__ AVFrame gpu_frame;

#define YUV_INTER 0
#define RGB_INTER 1
#define BLOCK_SIZE 32

char *intermediate;
char *gpu_out_buffer;
AVF_GPU_wrapper wrapper;

template<typename SRC, typename DEST>
__device__ void R10k_from_rgb(const SRC *src, DEST *dst1, int shift){
    uint32_t *dst = (uint32_t *) dst1;
    comp_type_t r = *src++ >> shift;
    comp_type_t g = *src++ >> shift;
    comp_type_t b = *src++ >> shift;
    // B5-B0 XX | G3-G0 B9-B6 | R1-R0 G9-G4 | R9-R2
    *dst = (b & 0x3FU) << 26U | 0x3000000U | (g & 0xFU) << 20U | (b >> 6U) << 16U | (r & 0x3U) << 14U | (g >> 4U) << 8U | r >> 2U;
}

template<typename SRC, typename DEST>
__device__ void RGB_from_rgb(const SRC *src, DEST *dst, bool has_alpha, int shift){
    *dst++ = *src++ >> shift;
    *dst++ = *src++ >> shift;
    *dst++ = *src++ >> shift;
    if (has_alpha)
        *dst++ = *src++ >> shift;
}

__device__ void rgb_to_r12l(char * __restrict dst, auto GET_NEXT)
{
    comp_type_t r, g, b;
    comp_type_t tmp;

    GET_NEXT( r, g, b); // 0
    dst[BYTE_SWAP(0)] = r & 0xFFU;
    dst[BYTE_SWAP(1)] = (g & 0xFU) << 4U | r >> 8U;
    dst[BYTE_SWAP(2)] = g >> 4U;
    dst[BYTE_SWAP(3)] = b & 0xFFU;
    tmp = b >> 8U;

    GET_NEXT( r, g, b); // 1
    dst[4 + BYTE_SWAP(0)] = (r & 0xFU) << 4U | tmp;
    dst[4 + BYTE_SWAP(1)] = r >> 4U;
    dst[4 + BYTE_SWAP(2)] = g & 0xFFU;
    dst[4 + BYTE_SWAP(3)] = (b & 0xFU) << 4U | g >> 8U;

    dst[8 + BYTE_SWAP(0)] = b >> 4U;
    GET_NEXT( r, g, b); // 2
    dst[8 + BYTE_SWAP(1)] = r & 0xFFu;
    dst[8 + BYTE_SWAP(2)] = (g & 0xFU) << 4U | r >> 8U;
    dst[8 + BYTE_SWAP(3)] = g >> 4U;

    dst[12 + BYTE_SWAP(0)] = b & 0xFFU;
    tmp = b >> 8U;
    GET_NEXT( r, g, b); // 3
    dst[12 + BYTE_SWAP(1)] = (r & 0xFU) << 4U | tmp;
    dst[12 + BYTE_SWAP(2)] = r >> 4U;
    dst[12 + BYTE_SWAP(3)] = g & 0xFFU;

    dst[16 + BYTE_SWAP(0)] = (b & 0xFU) << 4U | g >> 8U;
    dst[16 + BYTE_SWAP(1)] = b >> 4U;
    GET_NEXT( r, g, b); // 4
    dst[16 + BYTE_SWAP(2)] = r & 0xFFU;
    dst[16 + BYTE_SWAP(3)] = (g & 0xFU) << 4U | r >> 8U;

    dst[20 + BYTE_SWAP(0)] = g >> 4U;
    dst[20 + BYTE_SWAP(1)] = b & 0xFFU;
    tmp = b >> 8U;
    GET_NEXT( r, g, b); // 5
    dst[20 + BYTE_SWAP(2)] = (r & 0xFU) << 4U | tmp;
    dst[20 + BYTE_SWAP(3)] = r >> 4U;

    dst[24 + BYTE_SWAP(0)] = g & 0xFFU;
    dst[24 + BYTE_SWAP(1)] = (b & 0xFU) << 4U | g >> 8U;
    dst[24 + BYTE_SWAP(2)] = b >> 4U;
    GET_NEXT( r, g, b); // 6
    dst[24 + BYTE_SWAP(3)] = r & 0xFFU;

    dst[28 + BYTE_SWAP(0)] = (g & 0xFU) << 4U | r >> 8U;
    dst[28 + BYTE_SWAP(1)] = g >> 4U;
    dst[28 + BYTE_SWAP(2)] = b & 0xFFU;
    tmp = b >> 8U;
    GET_NEXT( r, g, b); // 7
    dst[28 + BYTE_SWAP(3)] = (r & 0xFU) << 4U | tmp;

    dst[32 + BYTE_SWAP(0)] = r >> 4U;
    dst[32 + BYTE_SWAP(1)] = g & 0xFFU;
    dst[32 + BYTE_SWAP(2)] = (b & 0xFU) << 4U | g >> 8U;
    dst[32 + BYTE_SWAP(3)] = b >> 4U;
}

__device__ void yuv_to_v210(uint32_t * __restrict d, auto GET_NEXT){
    comp_type_t y1, y2, u ,v;

    GET_NEXT(y1, y2, u, v);
    *d++ = u | y1 << 10 | v << 20;
    *d = y2;

    GET_NEXT(y1, y2, u, v);
    *d |= u << 10 | y1 << 20;
    *++d = v | y2 << 10;

    GET_NEXT(y1, y2, u, v);
    *d |= u << 20;
    *++d = y1 | v << 10 | y2 << 20;
}

template<typename DST, bool is_reversed>
__device__ void write_uyvy(DST *dst, auto GET_VALS){
    DST u, y0, v, y1;
    GET_VALS(u, y0, v, y1);

    if constexpr (!is_reversed){
        *dst++ =  u;
        *dst++ = y0;
        *dst++ =  v;
        *dst = y1;
    } else{
        *dst++ = y0;
        *dst++ = u;
        *dst++ = y1;
        *dst = v;
    }
}

/**************************************************************************************************************/
/*                                          KERNELS FROM YUV                                                  */
/**************************************************************************************************************/
template <typename IN_T, codec_t codec, int BIT_DEPTH>
__global__ void write_from_yuv_to_rgb(char * __restrict dst_buf, const char *__restrict src,
                                      int pitch, size_t pitch_in, int width, int height){

    size_t x = blockDim.x * blockIdx.x + threadIdx.x;
    size_t y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x >= width || y >= height )
        return;

    void *dst_row = dst_buf + pitch * y;

    const uint16_t *in = (uint16_t *) (src + y * pitch_in + 4 * 2 * x) ;
    comp_type_t y1, u, v, r, g, b, a;

    u = *in++ - (1<<15);
    y1 = Y_SCALE * (*in++ - (1<<12));
    v = *in++ - (1<<15);

    r = (YCBCR_TO_R_709_SCALED(y1, u, v) >> (COMP_BASE + 16-BIT_DEPTH));
    g = (YCBCR_TO_G_709_SCALED(y1, u, v) >> (COMP_BASE + 16-BIT_DEPTH));
    b = (YCBCR_TO_B_709_SCALED(y1, u, v) >> (COMP_BASE + 16-BIT_DEPTH));

    r = CLAMP_FULL(r, BIT_DEPTH);
    g = CLAMP_FULL(g, BIT_DEPTH);
    b = CLAMP_FULL(b, BIT_DEPTH);

    a = 0xFFU; //default alpha
    comp_type_t rgb[] = {r, g, b, a};

    if constexpr (codec == R10k){
        R10k_from_rgb((comp_type_t *) &rgb, ((IN_T *) dst_row) + 4 * x, 0);
    } else if constexpr (codec == RGB){
        RGB_from_rgb((comp_type_t *) &rgb, ((IN_T *) dst_row) + 3 * x, false, 0);
    } else if constexpr (codec == RGBA){
        rgb[3] = *in;
        RGB_from_rgb((comp_type_t *) &rgb, ((IN_T *) dst_row) + 4 * x, true, 0);
    } else if (codec == RG48){
        RGB_from_rgb((comp_type_t *) &rgb, ((IN_T *) dst_row) + 3 * x, false, 0);
    }
}

__global__ void write_yuv_to_r12l(char * __restrict dst_buf, const char *__restrict src_buf,
                                  int pitch, size_t pitch_in, int width, int height)
{
    size_t x = blockDim.x * blockIdx.x + threadIdx.x;
    size_t y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x >= width / 8 || y >= height)
        return;

    const void *src_row = src_buf + pitch_in * y;
    const uint16_t *src = ((const uint16_t *) src_row) + 8 * 4 * x;

    void * dst_row = dst_buf + pitch * y;
    char *dst = ((char *)dst_row) + 36 * x;

    auto GET_NEXT = [in=src](auto &r, auto &g, auto &b) mutable {
        comp_type_t y1, u, v;
        u = *in++ - (1 << 15);
        y1 = Y_SCALE * (*in++ - (1 << 12));
        v = *in++ - (1 << 15);
        in++;
        r = (YCBCR_TO_R_709_SCALED(y1, u, v) >> (COMP_BASE + 4U));
        g = (YCBCR_TO_G_709_SCALED(y1, u, v) >> (COMP_BASE + 4U));
        b = (YCBCR_TO_B_709_SCALED(y1, u, v) >> (COMP_BASE + 4U));
        r = CLAMP_FULL(r, 12);
        g = CLAMP_FULL(g, 12);
        b = CLAMP_FULL(b, 12);
    };
    rgb_to_r12l(dst, GET_NEXT);
}

template<bool is_reversed>
__global__ void write_from_yuv_to_uyvy(char * __restrict dst_buf, const char *__restrict src_buf,
                                       int pitch, size_t pitch_in, int width, int height){
    size_t x = blockDim.x * blockIdx.x + threadIdx.x;
    size_t y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x >= width / 2 || y >= height)
        return;

    const void *src_row = src_buf + pitch_in * y;
    const uint8_t *src = ((const uint8_t *) src_row) + 4 * 2 * 2 * x;

    void * dst_row = dst_buf + pitch * y;
    char *dst = ((char *) dst_row) + 4 * x;

    auto GET_VALS = [src](auto &u, auto &y0, auto &v, auto &y1){
        u = (src[1] + src[9]) / 2; // Uuint8_t
        y0 = src[3]; // Y0
        v = (src[5] + src[13]) / 2; // V
        y1 = src[11]; // Y1
    };

    write_uyvy<char, is_reversed>(dst, GET_VALS);
}

__global__ void write_from_yuv_to_v210(char * __restrict dst_buf, const char *__restrict src_buf,
                                       int pitch, size_t pitch_in, int width, int height){
    size_t x = blockDim.x * blockIdx.x + threadIdx.x;
    size_t y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x >= width / 6 || y >= height)
        return;

    const void *src_row = src_buf + pitch_in * y;
    const uint16_t *src = ((const uint16_t *) src_row) + 3 * 2 * 4 * x;

    void * dst_row = dst_buf + pitch * y;
    uint32_t *dst = ((uint32_t *)dst_row) + 4 * x;

    auto FETCH_BLOCK = [in = src](auto &y1, auto &y2, auto &u, auto &v) mutable {

        u = *in++ >> 6;
        y1 = *in++ >> 6;
        v = *in++ >> 6;
        in++;

        u += *in++ >> 6;
        y2 = *in++ >> 6;
        v += *in++ >> 6;
        in++;

        u = u / 2 ;
        v = v / 2 ;
    };
    yuv_to_v210(dst, FETCH_BLOCK);
}

__global__ void write_from_yuv_to_y216(char * __restrict dst_buf, const char *__restrict src_buf,
                                       int pitch, size_t pitch_in, int width, int height)
{
    size_t x = blockDim.x * blockIdx.x + threadIdx.x;
    size_t y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x >= width / 2 || y >= height)
        return;

    const void *src_row = src_buf + pitch_in * y;
    const uint16_t *src = ((const uint16_t *) src_row) + 4 * 2 * x;

    void * dst_row = dst_buf + pitch * y;
    uint16_t *dst = ((uint16_t *) dst_row) + 4 * x;

    auto GET_VALS = [src](auto &u, auto &y0, auto &v, auto &y1){
        u = (src[0] + src[4]) / 2; // U
        y0 = src[1]; // Y0
        v = (src[2] + src[6]) / 2; // V
        y1 = src[5]; // Y1
    };

    write_uyvy<uint16_t, true>(dst, GET_VALS);
}

/**************************************************************************************************************/
/*                                            KERNELS FROM RGB                                                */
/**************************************************************************************************************/


template <bool is_reversed>
__global__ void write_from_rgb_to_uyvy(char * __restrict dst_buf, const char *__restrict src_buf,
                                       int pitch, size_t pitch_in, int width, int height){

    size_t x = blockDim.x * blockIdx.x + threadIdx.x;
    size_t y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x >= width / 2 || y >= height)
        return;

    const void *src_row = src_buf + pitch_in * y;
    const uint16_t *src = ((const uint16_t *) src_row) + 4 * 2 * x;

    void * dst_row = dst_buf + pitch * y;
    char *dst = ((char *)dst_row) + 4 * x;

    auto GET_VALS = [src](auto &u, auto &y1, auto &v, auto &y2) mutable {
        int r, g, b, ty1, ty2, tu, tv;
        r = *src++ >> 8;
        g = *src++ >> 8;
        b = *src++ >> 8;
        src++;

        ty1 = 11993 * r + 40239 * g + 4063 * b + (1<<20);
        tu  = -6619 * r -22151 * g + 28770 * b;
        tv  = 28770 * r - 26149 * g - 2621 * b;

        r = *src++ >> 8;
        g = *src++ >> 8;
        b = *src++ >> 8;

        ty2 = 11993 * r + 40239 * g + 4063 * b + (1<<20);
        tu += -6619 * r -22151 * g + 28770 * b;
        tv += 28770 * r - 26149 * g - 2621 * b;

        tu = tu / 2 + (1<<23);
        tv = tv / 2 + (1<<23);

        u = CLAMP(tu, 0, (1<<24)-1) >> 16;
        v = CLAMP(tv, 0, (1<<24)-1) >> 16;
        y1 = CLAMP(ty1, 0, (1<<24)-1) >> 16;
        y2 = CLAMP(ty2, 0, (1<<24)-1) >> 16;
    };

    write_uyvy<char, is_reversed>(dst, GET_VALS);
}

template<typename OUT_T, codec_t codec>
__global__ void write_from_rgb_to_rgb(char * __restrict dst_buf, const char *__restrict src_buf,
                                      int pitch, size_t pitch_in, int width, int height){

    size_t x = blockDim.x * blockIdx.x + threadIdx.x;
    size_t y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x >= width || y >= height )
        return;

    const void *src_row = src_buf + pitch_in * y;
    const uint16_t *src = ((const uint16_t *) src_row) + 4 * x;

    void * dst_row = dst_buf + pitch * y;

    if constexpr (codec == R10k){
        R10k_from_rgb(src, ((OUT_T *) dst_row) + 4 * x, 6);
    } else if constexpr (codec == RGB){
        RGB_from_rgb(src, ((OUT_T *) dst_row) + 3 * x, false, 8);
    } else if constexpr (codec == RGBA){
        RGB_from_rgb(src, ((OUT_T *) dst_row) + 4 * x, true, 8);
    } else if constexpr (codec == RG48){
        RGB_from_rgb(src, ((OUT_T *) dst_row) + 3 * x, false, 0);
    }
}

__global__ void write_from_rgb_to_r12l(char * __restrict dst_buf, const char *__restrict src_buf,
                                       int pitch, size_t pitch_in, int width, int height){
    size_t x = blockDim.x * blockIdx.x + threadIdx.x;
    size_t y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x >= width / 8 || y >= height)
        return;

    const void *src_row = src_buf + pitch_in * y;
    const uint16_t *src = ((const uint16_t *) src_row) + 8 * 4 * x;

    void * dst_row = dst_buf + pitch * y;
    char *dst = ((char *)dst_row) + 36 * x;

    auto GET_NEXT = [in=src](auto &r, auto &g, auto &b) mutable {
        r = *in++ >> 4;
        g = *in++ >> 4;
        b = *in++ >> 4;
        in++;
    };
    rgb_to_r12l(dst, GET_NEXT);
}

__global__ void write_from_rgb_to_v210(char * __restrict dst_buf, const char *__restrict src_buf,
                                       int pitch, size_t pitch_in, int width, int height){
    size_t x = blockDim.x * blockIdx.x + threadIdx.x;
    size_t y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x >= width / 6 || y >= height)
        return;

    const void *src_row = src_buf + pitch_in * y;
    const uint16_t *src = ((const uint16_t *) src_row) + 3 * 2 * 4 * x;

    void * dst_row = dst_buf + pitch * y;
    uint32_t *dst = ((uint32_t *)dst_row) + 4 * x;

    auto FETCH_BLOCK = [in = src](auto &y1, auto &y2, auto &u, auto &v) mutable {
        comp_type_t r, g, b;

        r = *in++;
        g = *in++;
        b = *in++;
        in++;

        y1 = (RGB_TO_Y_709_SCALED(r, g, b) >> (COMP_BASE + (16 - 10))) + (1 << 6);
        u = RGB_TO_CB_709_SCALED(r, g, b) >> (COMP_BASE + (16 - 10));
        v = RGB_TO_CR_709_SCALED(r, g, b) >> (COMP_BASE + (16 - 10));

        r = *in++;
        g = *in++;
        b = *in++;
        in++;

        y2 = (RGB_TO_Y_709_SCALED(r, g, b) >> (COMP_BASE + (16 - 10))) + (1 << 6);
        u += RGB_TO_CB_709_SCALED(r, g, b) >> (COMP_BASE + (16 - 10));
        v += RGB_TO_CR_709_SCALED(r, g, b) >> (COMP_BASE + (16 - 10));

        u = u / 2 + (1 << 9);
        v = v / 2 + (1 << 9);

        y1 = CLAMP_LIMITED_Y(y1, 10);
        y2 = CLAMP_LIMITED_Y(y2, 10);
        u = CLAMP_LIMITED_CBCR(u, 10);
        v = CLAMP_LIMITED_CBCR(v, 10);
    };
    yuv_to_v210(dst, FETCH_BLOCK);
}


__global__ void write_from_rgb_to_y216(char * __restrict dst_buf, const char *__restrict src_buf,
                                       int pitch, size_t pitch_in, int width, int height){
    size_t x = blockDim.x * blockIdx.x + threadIdx.x;
    size_t y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x >= width / 2 || y >= height)
        return;

    const void *src_row = src_buf + pitch_in * y;
    const uint16_t *src = ((const uint16_t *) src_row) + 4 * 2 * x;

    void * dst_row = dst_buf + pitch * y;
    uint16_t *dst = ((uint16_t *) dst_row) + 4 * x;

    auto FETCH_BLOCK = [in = src](auto &u, auto &y1, auto &v, auto &y2) mutable {
        comp_type_t r, g, b;

        r = *in++;
        g = *in++;
        b = *in++;
        in++;

        y1 = (RGB_TO_Y_709_SCALED(r, g, b) >> COMP_BASE) + (1<<12);
        y1 = CLAMP_LIMITED_Y(y1, 16);

        u = (RGB_TO_CB_709_SCALED(r, g, b) >> COMP_BASE);
        v = (RGB_TO_CR_709_SCALED(r, g, b) >> COMP_BASE);

        r = *in++;
        g = *in++;
        b = *in++;
        in++;

        y2 = (RGB_TO_Y_709_SCALED(r, g, b) >> COMP_BASE) + (1<<12);
        y2 = CLAMP_LIMITED_Y(y2, 16);

        u = (u + (RGB_TO_CB_709_SCALED(r, g, b) >> COMP_BASE) / 2) + (1<<15);
        u = CLAMP_LIMITED_CBCR(u, 16);

        v = (v + (RGB_TO_CR_709_SCALED(r, g, b) >> COMP_BASE) / 2) + (1<<15);
        v = CLAMP_LIMITED_CBCR(v, 16);
    };

    write_uyvy<uint16_t, true>(dst, FETCH_BLOCK);
}
__global__ void write_from_rgb_to_y416(char * __restrict dst_buf, const char *__restrict src_buf,
                                       int pitch, size_t pitch_in, int width, int height){

    size_t x = blockDim.x * blockIdx.x + threadIdx.x;
    size_t y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    const void *src_row = src_buf + pitch_in * y;
    const uint16_t *src = ((const uint16_t *) src_row) + 4 * x;

    void * dst_row = dst_buf + pitch * y;
    uint16_t *dst = ((uint16_t *) dst_row) + 4 * x;

    comp_type_t r, g, b, y1, u, v;

    r = *src++;
    g = *src++;
    b = *src++;

    y1 = (RGB_TO_Y_709_SCALED(r, g, b) >> COMP_BASE)+ (1<<12);
    u = (RGB_TO_CB_709_SCALED(r, g, b) >> COMP_BASE)+ (1<<15);
    v = (RGB_TO_CR_709_SCALED(r, g, b) >> COMP_BASE)+ (1<<15);

    y1 = CLAMP_LIMITED_Y(y1, 16);
    u = CLAMP_LIMITED_CBCR(u, 16);
    v = CLAMP_LIMITED_CBCR(v, 16);

    *dst++ = u;
    *dst++ = y1;
    *dst++ = v;
    *dst   = *src;
}

/**************************************************************************************************************/
/*                                            KERNELS TO                                                      */
/**************************************************************************************************************/

__global__ void ayuv64_to_y416(char * __restrict dst_buffer, int pitch, int width, int height)
{
    //yuv444p -> yuv444i
    AVFrame *in_frame = &gpu_frame;
    size_t x = blockDim.x * blockIdx.x + threadIdx.x;
    size_t y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    void *src_row = in_frame->data[0] + in_frame->linesize[0] *  y;
    void *dst_row = dst_buffer + y * pitch;

    uint16_t * src = ((uint16_t *) src_row) + 4 * x;
    uint16_t * dst = ((uint16_t *) dst_row) + 4 * x;

    *dst++ = src[2]; // U
    *dst++ = src[1]; // Y
    *dst++ = src[3]; // V
    *dst = src[0]; // A
}

__global__ void xv30_to_intermediate(char * __restrict dst_buffer, int pitch, int width, int height){

    AVFrame *in_frame = &gpu_frame;
    size_t x = blockDim.x * blockIdx.x + threadIdx.x;
    size_t y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    void *src_row = in_frame->data[0] + in_frame->linesize[0] *  y;
    void *dst_row = dst_buffer + y * pitch;

    char * src = ((char *) src_row) + 4 * x;
    uint16_t * dst = ((uint16_t *) dst_row) + 4 * x;

    uint32_t in = *src;
    *dst++ = (in & 0x3FFU) << 6U;
    *dst++ = ((in >> 10U) & 0x3FFU) << 6U;
    *dst++ = ((in >> 20U) & 0x3FFU) << 6U;
    *dst   = 0xFFFFU;

}

template<typename IN_T, int bit_shift>
__global__ void p010le_to_inter(char * __restrict dst_buffer, int pitch, int width, int height)
{
    // y cbcr -> yuv 444 i
    AVFrame *in_frame = &gpu_frame;
    size_t x = blockDim.x * blockIdx.x + threadIdx.x;
    size_t y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x >= width /2 || y >= height/2)
        return;

    IN_T * src_cbcr, *src_y1, *src_y2;

    //y1, y2
    void * src_y1_row = in_frame->data[0] + in_frame->linesize[0] * 2 * y;
    void * src_y2_row = in_frame->data[0] + in_frame->linesize[0] * (2 * y + 1);
    src_y1 = ((IN_T *) (src_y1_row)) + 2 * x;
    src_y2 = ((IN_T *) (src_y2_row)) + 2 * x;

    void *src_cbcr_row = in_frame->data[1] + in_frame->linesize[1] * y;
    src_cbcr = ((IN_T *) src_cbcr_row) + 2 * x;

    //dst
    void *dst1_row = dst_buffer + (y * 2) * pitch;
    void *dst2_row = dst_buffer + (y * 2 +1) * pitch;
    uint16_t *dst1 = ((uint16_t *) dst1_row) + 4 * 2 * x;
    uint16_t *dst2 = ((uint16_t *) dst2_row) + 4 * 2 * x;

    uint16_t tmp;
    for (int _ = 0; _ < 2; ++_){
        // U
        tmp = src_cbcr[0] << bit_shift;
        *dst1++ = tmp;
        *dst2++ = tmp;
        // Y
        *dst1++ = *src_y1++ << bit_shift;
        *dst2++ = *src_y2++ << bit_shift;
        // V
        tmp = src_cbcr[1] << bit_shift;
        *dst1++ = tmp;
        *dst2++ = tmp;
        //A
        *dst1++ = 0xFFFFU;
        *dst2++ = 0xFFFFU;
    }
}

template<typename IN_T, int bit_shift>
__global__ void yuv422p_to_intermediate(char * __restrict dst_buffer, int pitch, int width, int height){
    // yuvp -> yuv 444 i
    AVFrame *in_frame = &gpu_frame;
    size_t x = blockDim.x * blockIdx.x + threadIdx.x;
    size_t y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x >= width /2 || y >= height/2)
        return;

    IN_T * src_cb1, *src_cb2, *src_cr1, *src_cr2, *src_y1, *src_y2;

    //y1, y2
    void * src_y1_row = in_frame->data[0] + in_frame->linesize[0] * 2 * y;
    void * src_y2_row = in_frame->data[0] + in_frame->linesize[0] * (2 * y + 1);
    src_y1 = ((IN_T *) (src_y1_row)) + 2 * x;
    src_y2 = ((IN_T *) (src_y2_row)) + 2 * x;

    //dst
    void * dst1_row = dst_buffer +  2 * y      * pitch;
    void * dst2_row = dst_buffer + (2 * y + 1) * pitch;
    uint16_t *dst1 = ((uint16_t *) dst1_row) + 4 * 2 * x;
    uint16_t *dst2 = ((uint16_t *) dst2_row) + 4 * 2 * x;

    void * src_cb1_row = in_frame->data[1] + in_frame->linesize[1] *  2 * y;
    void * src_cb2_row = in_frame->data[1] + in_frame->linesize[1] *  (2 * y + 1);

    void * src_cr1_row = in_frame->data[2] + in_frame->linesize[2] *  2 * y;
    void * src_cr2_row = in_frame->data[2] + in_frame->linesize[2] *  (2 * y + 1);

    src_cb1 = ((IN_T *) (src_cb1_row)) + x;
    src_cb2 = ((IN_T *) (src_cb2_row)) + x;

    src_cr1 = ((IN_T *) (src_cr1_row)) + x;
    src_cr2 = ((IN_T *) (src_cr2_row)) + x;

    for (int _ = 0; _ < 2; ++_) {
        *dst1++ = *src_cb1 << bit_shift; // U
        *dst1++ = *src_y1++ << bit_shift; // Y
        *dst1++ = *src_cr1 << bit_shift; // V
        *dst1++ = 0xFFFFU; // A
    }

    for (int _ = 0; _ < 2; ++_) {
        *dst2++ = *src_cb2 << bit_shift; // U
        *dst2++ = *src_y2++ << bit_shift; // Y
        *dst2++ = *src_cr2 << bit_shift; // V
        *dst2++ = 0xFFFFU; // A
    }
}

template<typename IN_T, int bit_shift>
__global__ void yuv420p_to_intermediate(char * __restrict dst_buffer, int pitch, int width, int height){
    // yuvp -> yuv 444 i
    AVFrame *in_frame = &gpu_frame;
    size_t x = blockDim.x * blockIdx.x + threadIdx.x;
    size_t y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x >= width /2 || y >= height/2)
        return;

    IN_T * src_cb1, *src_cr1, *src_y1, *src_y2;

    //y1, y2
    void * src_y1_row = in_frame->data[0] + in_frame->linesize[0] * 2 * y;
    void * src_y2_row = in_frame->data[0] + in_frame->linesize[0] * (2 * y + 1);
    src_y1 = ((IN_T *) (src_y1_row)) + 2 * x;
    src_y2 = ((IN_T *) (src_y2_row)) + 2 * x;

    //dst
    void * dst1_row = dst_buffer +  2 * y      * pitch;
    void * dst2_row = dst_buffer + (2 * y + 1) * pitch;
    uint16_t *dst1 = ((uint16_t *) dst1_row) + 4 * 2 * x;
    uint16_t *dst2 = ((uint16_t *) dst2_row) + 4 * 2 * x;

    //fills the same cr/cb to each line
    void * src_cb_row = in_frame->data[1] + in_frame->linesize[1] *  y;
    void * src_cr_row = in_frame->data[2] + in_frame->linesize[2] *  y;

    src_cb1 = ((IN_T *) (src_cb_row)) + x;
    src_cr1 = ((IN_T *) (src_cr_row)) + x;

    for (int _ = 0; _ < 2; ++_) {
        *dst1++ = *src_cb1 << bit_shift; // U
        *dst1++ = *src_y1++ << bit_shift; // Y
        *dst1++ = *src_cr1 << bit_shift; // V
        *dst1++ = 0xFFFFU; // A
    }
    for (int _ = 0; _ < 2; ++_) {
        *dst2++ = *src_cb1 << bit_shift; // U
        *dst2++ = *src_y2++ << bit_shift; // Y
        *dst2++ = *src_cr1 << bit_shift; // V
        *dst2++ = 0xFFFFU; // A
    }
}

template<typename IN_T, int bit_shift>
__global__ void yuv444p_to_intermediate(char * __restrict dst_buffer, int pitch, int width, int height){
    //yuv444p -> yuv444i
    AVFrame *in_frame = &gpu_frame;
    size_t x = blockDim.x * blockIdx.x + threadIdx.x;
    size_t y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    void *src_y1_row = in_frame->data[0] + in_frame->linesize[0] *  y;
    void *src_cb_row = in_frame->data[1] + in_frame->linesize[1] *  y;
    void *src_cr_row = in_frame->data[2] + in_frame->linesize[2] *  y;
    void *dst_row = dst_buffer + y * pitch;

    IN_T *   src_y1 = ((IN_T *) src_y1_row) + x;
    IN_T *   src_cb = ((IN_T *) src_cb_row) + x;
    IN_T *   src_cr = ((IN_T *) src_cr_row) + x;
    uint16_t * dst1 = ((uint16_t *) dst_row) + 4 * x;

    *dst1++ = *src_cb << bit_shift; // U
    *dst1++ = *src_y1 << bit_shift; // Y
    *dst1++ = *src_cr << bit_shift; // V
    *dst1++ = 0xFFFFU; // A
}

template<typename IN_T, int bit_shift, bool has_alpha>
__global__ void gbrap_to_intermediate(char * __restrict dst_buffer, int pitch, int width, int height)
{
    AVFrame *in_frame = &gpu_frame;
    size_t x = blockDim.x * blockIdx.x + threadIdx.x;
    size_t y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    void *src_g_row = in_frame->data[0] + in_frame->linesize[0] *  y;
    void *src_b_row = in_frame->data[1] + in_frame->linesize[1] *  y;
    void *src_r_row = in_frame->data[2] + in_frame->linesize[2] *  y;
    void *dst_row = dst_buffer + y * pitch;

    IN_T *src_g = ((IN_T *) src_g_row) + x;
    IN_T *src_b = ((IN_T *) src_b_row) + x;
    IN_T *src_r = ((IN_T *) src_r_row) + x;

    uint16_t *dst = ((uint16_t *) dst_row) + 4 * x;

    *dst++ = *src_r << bit_shift;
    *dst++ = *src_g << bit_shift;
    *dst++ = *src_b << bit_shift;
    if constexpr (has_alpha){
        void * src_a_row = in_frame->data[3] + in_frame->linesize[3] * y;
        IN_T *src_a =((IN_T *) src_a_row) + x;
        *dst = *src_a << bit_shift;
    } else{
        *dst = 0xFFFFU;
    }
}

template<typename IN_T, int bit_shift, bool has_alpha>
__global__ void rgb_to_intermediate(char * __restrict dst_buffer, int pitch, int width, int height)
{
    AVFrame *in_frame = &gpu_frame;
    size_t x = blockDim.x * blockIdx.x + threadIdx.x;
    size_t y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    void *src_row = in_frame->data[0] + in_frame->linesize[0] *  y;
    void *dst_row = dst_buffer + y * pitch;

    IN_T *src = ((IN_T *) src_row) + (has_alpha ? 4 : 3) * x;
    uint16_t *dst = ((uint16_t *) dst_row) + 4 * x;

    *dst++ = *src++ << bit_shift;
    *dst++ = *src++ << bit_shift;
    *dst++ = *src++ << bit_shift;
    if constexpr (has_alpha){
        *dst = *src;
    } else{
        *dst = 0xFFFFU;
    }
}

template<bool has_alpha>
__global__ void vuya_to_intermediate(char * __restrict dst_buffer, int pitch, int width, int height)
{
    AVFrame *in_frame = &gpu_frame;
    size_t x = blockDim.x * blockIdx.x + threadIdx.x;
    size_t y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    void *src_row = in_frame->data[0] + in_frame->linesize[0] *  y;
    void *dst_row = dst_buffer + y * pitch;

    char *src = ((char *) src_row) + (has_alpha ? 4 : 3) * x;
    uint16_t *dst = ((uint16_t *) dst_row) + 4 * x;

    *dst++ = src[1] << 8U;
    *dst++ = src[2] << 8U;
    *dst++ = src[0] << 8U;
    if constexpr (has_alpha)
        *dst = src[3] << 8U;
    else
        *dst = 0xFFFF;
}

__global__ void y210_to_intermediate(char * __restrict dst_buffer, int pitch, int width, int height){
    AVFrame *in_frame = &gpu_frame;
    size_t x = blockDim.x * blockIdx.x + threadIdx.x;
    size_t y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    void *src_row = in_frame->data[0] + in_frame->linesize[0] *  y;
    void *dst_row = dst_buffer + y * pitch;

    char *src = ((char *) src_row) + 4 * x;
    uint16_t *dst = ((uint16_t *) dst_row) + 4 * 2 * x;

    unsigned y0, u, y1, v;
    y0 = *src++;
    u = *src++;
    y1 = *src++;
    v = *src;

    *dst++ = u << 8;
    *dst++ = y0 << 8;
    *dst++ = v << 8;
    *dst++ = 0xFFFFU;

    *dst++ = u << 8;
    *dst++ = y1 << 8;
    *dst++ = v << 8;
    *dst   = 0xFFFFU;
}

__global__ void p210_to_inter(char * __restrict dst_buffer, int pitch, int width, int height){
    AVFrame *in_frame = &gpu_frame;
    size_t x = blockDim.x * blockIdx.x + threadIdx.x;
    size_t y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x >= width / 2 || y >= height)
        return;

    void *src_y_row = in_frame->data[0] + in_frame->linesize[0] * y;
    void *src_cbcr_row = in_frame->data[1] + in_frame->linesize[1] * y;
    void *dst_row = dst_buffer + y * pitch;

    uint16_t *src_cbcr = ((uint16_t *) src_cbcr_row) + 2 * x;
    uint16_t *src_y = ((uint16_t *) src_y_row) + 2 * x;
    uint16_t *dst = ((uint16_t *) dst_row) + 2 * 4 * x;

    uint16_t cb, cr;
    cb = *src_cbcr++;
    cr = *src_cbcr;

    *dst = cb;
    *dst++ = *src_y++;
    *dst++ = cr;
    *dst++ =  0xFFFFU;

    *dst = cb;
    *dst++ = *src_y;
    *dst++ = cr;
    *dst   =  0xFFFFU;
}

/**************************************************************************************************************/
/*                                              YUV FROM                                                      */
/**************************************************************************************************************/

template<typename T, codec_t codec, int d>
void convert_from_yuv_inter_to_rgb(const AVFrame *frame)
{
    size_t width = frame->width;
    size_t height = frame->height;
    size_t pitch = vc_get_linesize(width, codec);
    size_t pitch_in = vc_get_linesize(width, Y416);

    assert((uintptr_t) gpu_out_buffer % 4 == 0);
    assert((uintptr_t) frame->linesize[0] % 2 == 0); // Y
    assert((uintptr_t) frame->linesize[1] % 2 == 0); // U
    assert((uintptr_t) frame->linesize[2] % 2 == 0); // V

    //execute the conversion
    dim3 grid = dim3((width + BLOCK_SIZE - 1) / BLOCK_SIZE, (height + BLOCK_SIZE - 1) / BLOCK_SIZE );
    dim3 block = dim3(BLOCK_SIZE, BLOCK_SIZE);

    write_from_yuv_to_rgb<T, codec, d><<<grid, block>>>(gpu_out_buffer, intermediate, pitch, pitch_in, width, height);
}

template <bool IS_REVERSED>
void convert_from_yuv_to_uyvy(const AVFrame *frame){
    size_t width = frame->width;
    size_t height = frame->height;
    size_t pitch = vc_get_linesize(width, UYVY);
    size_t pitch_in = vc_get_linesize(width, Y416);

    dim3 grid = dim3((width/2 + BLOCK_SIZE - 1) / BLOCK_SIZE, (height + BLOCK_SIZE - 1) / BLOCK_SIZE );
    dim3 block = dim3(BLOCK_SIZE, BLOCK_SIZE);
    write_from_yuv_to_uyvy<IS_REVERSED><<<grid, block>>>(gpu_out_buffer, intermediate, pitch, pitch_in, width, height);
}

void convert_from_yuv_to_r12l(const AVFrame  *frame){
    size_t width = frame->width;
    size_t height = frame->height;
    size_t pitch = vc_get_linesize(width, R12L);
    size_t pitch_in = vc_get_linesize(width, Y416);

    dim3 grid = dim3((width / 8 + BLOCK_SIZE - 1) / BLOCK_SIZE, (height + BLOCK_SIZE - 1) / BLOCK_SIZE );
    dim3 block = dim3(BLOCK_SIZE, BLOCK_SIZE);
    write_yuv_to_r12l<<<grid, block>>>(gpu_out_buffer, intermediate, pitch, pitch_in, width, height);
}

void convert_from_yuv_to_v210(const AVFrame  *frame){
    size_t width = frame->width;
    size_t height = frame->height;
    size_t pitch = vc_get_linesize(width, v210);
    size_t pitch_in = vc_get_linesize(width, Y416);

    dim3 grid = dim3((width / 6 + BLOCK_SIZE - 1) / BLOCK_SIZE, (height + BLOCK_SIZE - 1) / BLOCK_SIZE );
    dim3 block = dim3(BLOCK_SIZE, BLOCK_SIZE);
    write_from_yuv_to_v210<<<grid, block>>>(gpu_out_buffer, intermediate, pitch, pitch_in, width, height);
}

void convert_from_yuv_to_y216(const AVFrame  *frame){
    size_t width = frame->width;
    size_t height = frame->height;
    size_t pitch = vc_get_linesize(width, Y216);
    size_t pitch_in = vc_get_linesize(width, Y416);

    dim3 grid = dim3((width / 2 + BLOCK_SIZE - 1) / BLOCK_SIZE, (height + BLOCK_SIZE - 1) / BLOCK_SIZE );
    dim3 block = dim3(BLOCK_SIZE, BLOCK_SIZE);
    write_from_yuv_to_y216<<<grid, block>>>(gpu_out_buffer, intermediate, pitch, pitch_in, width, height);
}

void convert_from_yuv_to_y416(const AVFrame *frame){
    size_t width = frame->width;
    size_t height = frame->height;
    cudaMemcpy((void *)gpu_out_buffer, (void *) intermediate, vc_get_datalen(width, height, Y416), cudaMemcpyDeviceToDevice);
}

/**************************************************************************************************************/
/*                                              RGB FROM                                                      */
/**************************************************************************************************************/

template<typename T, codec_t out>
void convert_from_rgb_inter_to_rgb(const AVFrame *frame){
    size_t width = frame->width;
    size_t height = frame->height;
    size_t pitch = vc_get_linesize(width, out);

    assert((uintptr_t) gpu_out_buffer % 4 == 0);
    assert((uintptr_t) frame->linesize[0] % 2 == 0); // Y
    assert((uintptr_t) frame->linesize[1] % 2 == 0); // U
    assert((uintptr_t) frame->linesize[2] % 2 == 0); // V

    //execute the conversion
    dim3 grid = dim3((width + BLOCK_SIZE - 1) / BLOCK_SIZE, (height + BLOCK_SIZE - 1) / BLOCK_SIZE );
    dim3 block = dim3(BLOCK_SIZE, BLOCK_SIZE);

    write_from_rgb_to_rgb<T, out>
    <<<grid, block>>>(gpu_out_buffer, intermediate, pitch, vc_get_linesize(width, Y416), width, height);
}

template<bool is_reversed>
void convert_from_rgb_to_uyvy(const AVFrame *frame){
    size_t width = frame->width;
    size_t height = frame->height;
    size_t pitch = vc_get_linesize(width, UYVY);
    size_t pitch_in = vc_get_linesize(width, Y416);

    dim3 grid = dim3((width + BLOCK_SIZE - 1) / BLOCK_SIZE, (height + BLOCK_SIZE - 1) / BLOCK_SIZE );
    dim3 block = dim3(BLOCK_SIZE, BLOCK_SIZE);
    write_from_rgb_to_uyvy<is_reversed><<<grid, block>>>(gpu_out_buffer, intermediate, pitch, pitch_in, width, height);
}

void convert_from_rgb_to_r12l(const AVFrame  *frame){
    size_t width = frame->width;
    size_t height = frame->height;
    size_t pitch = vc_get_linesize(width, R12L);
    size_t pitch_in = vc_get_linesize(width, Y416);

    dim3 grid = dim3((width / 6 + BLOCK_SIZE - 1) / BLOCK_SIZE, (height + BLOCK_SIZE - 1) / BLOCK_SIZE );
    dim3 block = dim3(BLOCK_SIZE, BLOCK_SIZE);
    write_from_rgb_to_r12l<<<grid, block>>>(gpu_out_buffer, intermediate, pitch, pitch_in, width, height);
}

void convert_from_rgb_to_v210(const AVFrame  *frame){
    size_t width = frame->width;
    size_t height = frame->height;
    size_t pitch = vc_get_linesize(width, v210);
    size_t pitch_in = vc_get_linesize(width, Y416);

    dim3 grid = dim3((width / 6 + BLOCK_SIZE - 1) / BLOCK_SIZE, (height + BLOCK_SIZE - 1) / BLOCK_SIZE );
    dim3 block = dim3(BLOCK_SIZE, BLOCK_SIZE);
    write_from_rgb_to_v210<<<grid, block>>>(gpu_out_buffer, intermediate, pitch, pitch_in, width, height);
}

void convert_from_rgb_to_y216(const AVFrame  *frame){
    size_t width = frame->width;
    size_t height = frame->height;
    size_t pitch = vc_get_linesize(width, Y216);
    size_t pitch_in = vc_get_linesize(width, Y416);

    dim3 grid = dim3((width / 2 + BLOCK_SIZE - 1) / BLOCK_SIZE, (height + BLOCK_SIZE - 1) / BLOCK_SIZE );
    dim3 block = dim3(BLOCK_SIZE, BLOCK_SIZE);
    write_from_rgb_to_y216<<<grid, block>>>(gpu_out_buffer, intermediate, pitch, pitch_in, width, height);
}

void convert_from_rgb_to_y416(const AVFrame *frame){
    size_t width = frame->width;
    size_t height = frame->height;
    size_t pitch = vc_get_linesize(width, Y416);
    size_t pitch_in = vc_get_linesize(width, Y416);

    dim3 grid = dim3((width + BLOCK_SIZE - 1) / BLOCK_SIZE, (height + BLOCK_SIZE - 1) / BLOCK_SIZE );
    dim3 block = dim3(BLOCK_SIZE, BLOCK_SIZE);
    write_from_rgb_to_y416<<<grid, block>>>(gpu_out_buffer, intermediate, pitch, pitch_in, width, height);
}

/**************************************************************************************************************/
/*                                                YUV TO                                                      */
/**************************************************************************************************************/

template<typename T, int i>
int convert_p010le_to_inter(const AVFrame *frame){
    size_t width = frame->width;
    size_t height = frame->height;
    size_t pitch = vc_get_linesize(width, Y416);

    //execute the conversion
    dim3 grid = dim3((width + BLOCK_SIZE - 1) / BLOCK_SIZE, (height + BLOCK_SIZE - 1) / BLOCK_SIZE );
    dim3 block = dim3(BLOCK_SIZE, BLOCK_SIZE);

    p010le_to_inter<T, i><<<grid, block>>>(intermediate, pitch, width, height);
    return YUV_INTER;
}

int convert_ayuv64_to_y416(const AVFrame *frame){
    size_t width = frame->width;
    size_t height = frame->height;
    size_t pitch = vc_get_linesize(width, Y416);

    //execute the conversion
    dim3 grid = dim3((width + BLOCK_SIZE - 1) / BLOCK_SIZE, (height + BLOCK_SIZE - 1) / BLOCK_SIZE );
    dim3 block = dim3(BLOCK_SIZE, BLOCK_SIZE);

    ayuv64_to_y416<<<grid, block>>>(intermediate, pitch, width, height);
    return YUV_INTER;
}

template<typename T, int bit_shift>
int convert_yuv422p_to_inter(const AVFrame *frame){
    size_t width = frame->width;
    size_t height = frame->height;
    size_t pitch = vc_get_linesize(width, Y416);

    assert((uintptr_t) gpu_out_buffer % 4 == 0);
    assert((uintptr_t) frame->linesize[0] % 2 == 0); // Y
    assert((uintptr_t) frame->linesize[1] % 2 == 0); // U
    assert((uintptr_t) frame->linesize[2] % 2 == 0); // V

    //execute the conversion
    dim3 grid;
    dim3 block = dim3(BLOCK_SIZE, BLOCK_SIZE);

    grid = dim3((width / 2 + BLOCK_SIZE - 1) / BLOCK_SIZE, (height / 2 + BLOCK_SIZE - 1) / BLOCK_SIZE );
    yuv422p_to_intermediate<T, bit_shift><<<grid, block>>>(intermediate, pitch, width, height);

    return YUV_INTER;
}

template<typename T, int bit_shift>
int convert_yuv420p_to_inter(const AVFrame *frame){
    size_t width = frame->width;
    size_t height = frame->height;
    size_t pitch = vc_get_linesize(width, Y416);

    assert((uintptr_t) gpu_out_buffer % 4 == 0);
    assert((uintptr_t) frame->linesize[0] % 2 == 0); // Y
    assert((uintptr_t) frame->linesize[1] % 2 == 0); // U
    assert((uintptr_t) frame->linesize[2] % 2 == 0); // V

    //execute the conversion
    dim3 grid;
    dim3 block = dim3(BLOCK_SIZE, BLOCK_SIZE);

    grid = dim3((width / 2 + BLOCK_SIZE - 1) / BLOCK_SIZE, (height / 2 + BLOCK_SIZE - 1) / BLOCK_SIZE );
    yuv420p_to_intermediate<T, bit_shift><<<grid, block>>>(intermediate, pitch, width, height);

    return YUV_INTER;
}

template<typename T, int bit_shift>
int convert_yuv444p_to_inter(const AVFrame *frame){
    size_t width = frame->width;
    size_t height = frame->height;
    size_t pitch = vc_get_linesize(width, Y416);

    assert((uintptr_t) gpu_out_buffer % 4 == 0);
    assert((uintptr_t) frame->linesize[0] % 2 == 0); // Y
    assert((uintptr_t) frame->linesize[1] % 2 == 0); // U
    assert((uintptr_t) frame->linesize[2] % 2 == 0); // V

    //execute the conversion
    dim3 grid;
    dim3 block = dim3(BLOCK_SIZE, BLOCK_SIZE);

    grid = dim3((width + BLOCK_SIZE - 1) / BLOCK_SIZE, (height + BLOCK_SIZE - 1) / BLOCK_SIZE );
    yuv444p_to_intermediate<T, bit_shift><<<grid, block>>>(intermediate, pitch, width, height);

    return YUV_INTER;
}

template<bool has_alpha>
int convert_yuva_to_inter(const AVFrame *frame){
    size_t width = frame->width;
    size_t height = frame->height;
    size_t pitch = vc_get_linesize(width, Y416);

    //execute the conversion
    dim3 grid = dim3((width + BLOCK_SIZE - 1) / BLOCK_SIZE, (height + BLOCK_SIZE - 1) / BLOCK_SIZE );
    dim3 block = dim3(BLOCK_SIZE, BLOCK_SIZE);

    vuya_to_intermediate<has_alpha><<<grid, block>>>(intermediate, pitch, width, height);
    return YUV_INTER;
}

int convert_y210_to_inter(const AVFrame *frame){
    size_t width = frame->width;
    size_t height = frame->height;
    size_t pitch = vc_get_linesize(width, Y416);

    //execute the conversion
    dim3 grid = dim3((width / 2 + BLOCK_SIZE - 1) / BLOCK_SIZE, (height + BLOCK_SIZE - 1) / BLOCK_SIZE );
    dim3 block = dim3(BLOCK_SIZE, BLOCK_SIZE);

    y210_to_intermediate<<<grid, block>>>(intermediate, pitch, width, height);
    return YUV_INTER;
}

int convert_xv30_to_inter(const AVFrame  *frame){
    size_t width = frame->width;
    size_t height = frame->height;
    size_t pitch = vc_get_linesize(width, Y416);

    //execute the conversion
    dim3 grid = dim3((width + BLOCK_SIZE - 1) / BLOCK_SIZE, (height + BLOCK_SIZE - 1) / BLOCK_SIZE );
    dim3 block = dim3(BLOCK_SIZE, BLOCK_SIZE);

    y210_to_intermediate<<<grid, block>>>(intermediate, pitch, width, height);
    return YUV_INTER;
}

int convert_p210_to_inter(const AVFrame  *frame){
    size_t width = frame->width;
    size_t height = frame->height;
    size_t pitch = vc_get_linesize(width, Y416);

    //execute the conversion
    dim3 grid = dim3((width / 2 + BLOCK_SIZE - 1) / BLOCK_SIZE, (height + BLOCK_SIZE - 1) / BLOCK_SIZE );
    dim3 block = dim3(BLOCK_SIZE, BLOCK_SIZE);

    p210_to_inter<<<grid, block>>>(intermediate, pitch, width, height);
    return YUV_INTER;
}

/**************************************************************************************************************/
/*                                               RGB TO                                                       */
/**************************************************************************************************************/

template<typename T, int bit_shift, bool has_alpha>
int convert_grb_to_inter(const AVFrame *frame){
    size_t width = frame->width;
    size_t height = frame->height;
    size_t pitch = vc_get_linesize(width, Y416);

    dim3 grid = dim3((width + BLOCK_SIZE - 1) / BLOCK_SIZE, (height + BLOCK_SIZE - 1) / BLOCK_SIZE );
    dim3 block = dim3(BLOCK_SIZE, BLOCK_SIZE);

    gbrap_to_intermediate<T, bit_shift, has_alpha><<<grid, block>>>(intermediate, pitch, width, height);
    return RGB_INTER;
}

template<typename T, int bit_shift, bool has_alpha>
int convert_rgb_to_inter(const AVFrame *frame){
    size_t width = frame->width;
    size_t height = frame->height;
    size_t pitch = vc_get_linesize(width, Y416);

    dim3 grid = dim3((width + BLOCK_SIZE - 1) / BLOCK_SIZE, (height + BLOCK_SIZE - 1) / BLOCK_SIZE );
    dim3 block = dim3(BLOCK_SIZE, BLOCK_SIZE);

    rgb_to_intermediate<T, bit_shift, has_alpha><<<grid, block>>>(intermediate, pitch, width, height);
    return RGB_INTER;
}

/**************************************************************************************************************/
/*                                                LIST                                                        */
/**************************************************************************************************************/

const std::map<int, int (*) (const AVFrame *)> conversions_to_inter = {
        // 10-bit YUV
        {AV_PIX_FMT_YUV420P10LE, convert_yuv420p_to_inter<uint16_t, 6>},
        {AV_PIX_FMT_YUV444P10LE, convert_yuv444p_to_inter<uint16_t, 6>},
        {AV_PIX_FMT_YUV422P10LE, convert_yuv422p_to_inter<uint16_t, 6>},
        {AV_PIX_FMT_P010LE, convert_p010le_to_inter<uint16_t, 0>},

        // 8-bit YUV (NV12)
        {AV_PIX_FMT_NV12, convert_p010le_to_inter<uint8_t, 8>},

        {AV_PIX_FMT_YUV420P, convert_yuv420p_to_inter<uint8_t, 8>},
        {AV_PIX_FMT_YUV422P, convert_yuv422p_to_inter<uint8_t, 8>},
        {AV_PIX_FMT_YUV444P, convert_yuv444p_to_inter<uint8_t, 8>},

        {AV_PIX_FMT_YUVJ420P, convert_yuv420p_to_inter<uint8_t, 8>},
        {AV_PIX_FMT_YUVJ422P, convert_yuv422p_to_inter<uint8_t, 8>},
        {AV_PIX_FMT_YUVJ444P, convert_yuv444p_to_inter<uint8_t, 8>},
        // 12-bit YUV
        {AV_PIX_FMT_YUV420P12LE, convert_yuv420p_to_inter<uint16_t, 4>},
        {AV_PIX_FMT_YUV422P12LE, convert_yuv422p_to_inter<uint16_t, 4>},
        {AV_PIX_FMT_YUV444P12LE, convert_yuv444p_to_inter<uint16_t, 4>},
        // 16-bit YUV
        {AV_PIX_FMT_YUV420P16LE, convert_yuv420p_to_inter<uint16_t , 0>},
        {AV_PIX_FMT_YUV422P16LE, convert_yuv422p_to_inter<uint16_t , 0>},
        {AV_PIX_FMT_YUV444P16LE, convert_yuv444p_to_inter<uint16_t , 0>},

        {AV_PIX_FMT_AYUV64, convert_ayuv64_to_y416},

        //GBR
        {AV_PIX_FMT_GBRP, convert_grb_to_inter<uint8_t, 8, false>},
        {AV_PIX_FMT_GBRAP, convert_grb_to_inter<uint8_t, 8, true>},

        {AV_PIX_FMT_GBRP10LE, convert_grb_to_inter<uint16_t, 6, false>},
        {AV_PIX_FMT_GBRP12LE, convert_grb_to_inter<uint16_t, 4, false>},
        {AV_PIX_FMT_GBRP16LE, convert_grb_to_inter<uint16_t, 0, false>},

        {AV_PIX_FMT_GBRAP10LE, convert_grb_to_inter<uint16_t, 6, true>},
        {AV_PIX_FMT_GBRAP12LE, convert_grb_to_inter<uint16_t, 4, true>},
        {AV_PIX_FMT_GBRAP16LE, convert_grb_to_inter<uint16_t, 0, true>},

        //RGB
        {AV_PIX_FMT_RGB24, convert_rgb_to_inter<uint8_t, 8, false>},
        {AV_PIX_FMT_RGB48LE, convert_rgb_to_inter<uint16_t, 0, false>},

        {AV_PIX_FMT_RGBA64LE, convert_rgb_to_inter<uint16_t, 0, true>},
        {AV_PIX_FMT_RGBA, convert_rgb_to_inter<uint8_t, 8, true>},

        {AV_PIX_FMT_Y210, convert_y210_to_inter}, //idk how to test these
#if P210_PRESENT
        {AV_PIX_FMT_P210LE, convert_p210_to_inter},
#endif
#if XV3X_PRESENT
        {AV_PIX_FMT_XV30, convert_xv30_to_inter}, //idk how to test these
        {AV_PIX_FMT_Y212, convert_y210_to_inter}, //idk how to test these
#endif
#if VUYX_PRESENT
        {AV_PIX_FMT_VUYA, convert_yuva_to_inter<true>}, //idk how to test these
        {AV_PIX_FMT_VUYX, convert_yuva_to_inter<false>} //idk how to test these
#endif
};

const std::map<int, void (*) (const AVFrame *frame)> conversions_from_yuv_inter = {
        {R10k, convert_from_yuv_inter_to_rgb<char, R10k, 10>},
        {RGB, convert_from_yuv_inter_to_rgb<char, RGB, 8>},
        {RGBA, convert_from_yuv_inter_to_rgb<char, RGBA, 8>},
        {RG48, convert_from_yuv_inter_to_rgb<uint16_t, RG48, 16>},

        {UYVY, convert_from_yuv_to_uyvy<false>},
        {YUYV, convert_from_yuv_to_uyvy<true>},
        {R12L, convert_from_yuv_to_r12l},
        {v210, convert_from_yuv_to_v210},
        {Y216, convert_from_yuv_to_y216},
        {Y416, convert_from_yuv_to_y416}
};

const std::map<int, void (*) (const AVFrame *)> conversions_from_rgb_inter = {
        {R10k, convert_from_rgb_inter_to_rgb<char, R10k>},
        {RGB, convert_from_rgb_inter_to_rgb<char, RGB>},
        {RGBA, convert_from_rgb_inter_to_rgb<char, RGBA>},
        {RG48, convert_from_rgb_inter_to_rgb<uint16_t, RG48>},

        {UYVY, convert_from_rgb_to_uyvy<false>},
        {YUYV, convert_from_rgb_to_uyvy<true>}, //idk how to test it ... but should work ig
        {R12L, convert_from_rgb_to_r12l},
        {v210, convert_from_rgb_to_v210},
        {Y216, convert_from_rgb_to_y216},
        {Y416, convert_from_rgb_to_y416}
};

/**************************************************************************************************************/
/*                                              INTERFACE                                                     */
/**************************************************************************************************************/

bool convert_from_lavc( const AVFrame* frame, char *dst, codec_t to) {
    wrapper.copy_to_device(frame);

    //copy host avframe struct to device
    cudaMemcpyToSymbol(gpu_frame, &(wrapper.frame), sizeof(AVFrame));
    auto converter_to = conversions_to_inter.at(frame->format);
    auto format = converter_to(frame);

    if (format == YUV_INTER){
        auto converter_from = conversions_from_yuv_inter.at(to);
        converter_from(frame);
    } else if (format == RGB_INTER){
        auto converter_from = conversions_from_rgb_inter.at(to);
        converter_from(frame);
    } else {
        //error
        std::cout << "error";
    }
    //copy the converted image back to the host
    cudaMemcpy(dst, gpu_out_buffer, vc_get_datalen(frame->width, frame->height, to), cudaMemcpyDeviceToHost);
    return true;
}

bool from_lavc_init(const AVFrame* frame, codec_t out, char **dst_ptr){
    if ( frame == nullptr || conversions_to_inter.find(frame->format) == conversions_to_inter.end()
         || conversions_from_rgb_inter.find(out) == conversions_from_rgb_inter.end()){ //both should contain same keys
        std::cout << "[from_lavc_converter] conversion not supported\n";
        return false;
    }
    cudaMalloc(&intermediate, vc_get_datalen(frame->width, frame->height, Y416));
    cudaMalloc(&gpu_out_buffer, vc_get_datalen(frame->width, frame->height, out));
    cudaMallocHost(dst_ptr, vc_get_datalen(frame->width, frame->height, out));
//    *dst_ptr = (char *) malloc(vc_get_datalen(frame->width, frame->height, out));

    wrapper.alloc(frame);

    return true;
}

void from_lavc_destroy(char *ptr){
    cudaFreeHost(ptr);
    cudaFree(intermediate);
    cudaFree(gpu_out_buffer);

    wrapper.free_from_device();
}
