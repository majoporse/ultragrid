#include "to_lavc.h"
#include "conv_utils.h"
#include <map>
#include <iostream>
#include "../src/color.h"

#define YUV_INTER_TO 0
#define RGB_INTER_TO 1

__device__ AVFrame gpu_frame;
char *intermediate_to;
char *gpu_in_buffer;

AVF_HOST_wrapper host_wrapper;
AVF_GPU_wrapper gpu_wrapper;

#define BLOCK_SIZE 32

/**************************************************************************************************************/
/*                                            KERNELS FROM                                                    */
/**************************************************************************************************************/

template< typename OUT_T, int bit_shift, bool has_alpha>
__global__ void convert_rgbp_from_inter(int width, int height, int pitch_in, char *in){
    //RGBA 16bit -> rgb AVF
    AVFrame *frame = &gpu_frame;
    size_t x = blockDim.x * blockIdx.x + threadIdx.x;
    size_t y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    void *dst_g_row = frame->data[0] + frame->linesize[0] *  y;
    void *dst_b_row = frame->data[1] + frame->linesize[1] *  y;
    void *dst_r_row = frame->data[2] + frame->linesize[2] *  y;
    void *src_row = in + y * pitch_in;

    OUT_T *dst_g = ((OUT_T *) dst_g_row) + x;
    OUT_T *dst_b = ((OUT_T *) dst_b_row) + x;
    OUT_T *dst_r = ((OUT_T *) dst_r_row) + x;

    uint16_t *src = ((uint16_t *) src_row) + 4 * x;

    *dst_r = *src++ >> bit_shift;
    *dst_g = *src++ >> bit_shift;
    *dst_b = *src++ >> bit_shift;
    if constexpr (has_alpha){
        void * dst_a_row = frame->data[3] + frame->linesize[3] * y;
        OUT_T *dst_a =((OUT_T *) dst_a_row) + x;
        *dst_a = *src >> bit_shift;
    }
}

__device__ void write_from_r12l(const uint8_t *src, auto WRITE_RES){

    comp_type_t r, g, b;

    r = (src[BYTE_SWAP(1)] & 0xFU) << 12U | src[BYTE_SWAP(0)] << 4U;                        //0
    g = src[BYTE_SWAP(2)] << 8U | (src[BYTE_SWAP(1)] & 0xF0U);
    b = (src[4 + BYTE_SWAP(0)] & 0xFU) << 12U | src[BYTE_SWAP(3)] << 4U;
    WRITE_RES(r, g, b);
    r = src[4 + BYTE_SWAP(1)] << 8U | (src[4 + BYTE_SWAP(0)] & 0xF0U);                      //1
    g = (src[4 + BYTE_SWAP(3)] & 0xFU) << 12U | (src[4 + BYTE_SWAP(2)]) << 4U;
    b = src[8 + BYTE_SWAP(0)] << 8U | (src[4 + BYTE_SWAP(3)] & 0xF0U);
    WRITE_RES(r, g, b);
    r = (src[8 + BYTE_SWAP(2)] & 0xFU) << 12U |src[8 + BYTE_SWAP(1)] << 4U;                 //2
    g = src[8 + BYTE_SWAP(3)] << 8U | (src[8 + BYTE_SWAP(2)] & 0xF0U);
    b = (src[12 + BYTE_SWAP(1)] & 0xFU) << 12U | src[12 + BYTE_SWAP(0)] << 4U;
    WRITE_RES(r, g, b);
    r = src[12 + BYTE_SWAP(2)] << 8U | (src[12 + BYTE_SWAP(1)] & 0xF0U);                    //3
    g = (src[16 + BYTE_SWAP(0)] & 0xFU) << 12U | src[12 + BYTE_SWAP(3)] << 4U;
    b = src[16 + BYTE_SWAP(1)] << 8U | (src[16 + BYTE_SWAP(0)] & 0xF0U);
    WRITE_RES(r, g, b);
    r = (src[16 + BYTE_SWAP(3)] & 0xFU) << 12U | src[16 + BYTE_SWAP(2)] << 4U;              //4
    g = src[20 + BYTE_SWAP(0)] << 8U | (src[16 + BYTE_SWAP(3)] & 0xF0U);
    b = (src[20 + BYTE_SWAP(2)] & 0xFU) << 12U | src[20 + BYTE_SWAP(1)] << 4U;
    WRITE_RES(r, g, b);
    r = src[20 + BYTE_SWAP(3)] << 8U | (src[20 + BYTE_SWAP(2)] & 0xF0U);                    //5
    g = (src[24 + BYTE_SWAP(1)] & 0xFU) << 12U | src[24 + BYTE_SWAP(0)] << 4U;
    b = src[24 + BYTE_SWAP(2)] << 8U | (src[24 + BYTE_SWAP(1)] & 0xF0U);
    WRITE_RES(r, g, b);
    r = (src[28 + BYTE_SWAP(0)] & 0xFU) << 12U | src[24 + BYTE_SWAP(3)] << 4U;              //6
    g = src[28 + BYTE_SWAP(1)] << 8U | (src[28 + BYTE_SWAP(0)] & 0xF0U);
    b = (src[28 + BYTE_SWAP(3)] & 0xFU) << 12U | src[28 + BYTE_SWAP(2)] << 4U;
    WRITE_RES(r, g, b);
    r = src[32 + BYTE_SWAP(0)] << 8U | (src[28 + BYTE_SWAP(3)] & 0xF0U);                    //7
    g = (src[32 + BYTE_SWAP(2)] & 0xFU) << 12U | src[32 + BYTE_SWAP(1)] << 4U;
    b = src[32 + BYTE_SWAP(3)] << 8U | (src[32 + BYTE_SWAP(2)] & 0xF0U);
    WRITE_RES(r, g, b);
}

__device__ void write_from_v210(const uint32_t *src, auto WRITE_RES){
    uint32_t w0_0 = *src++;
    uint32_t w0_1 = *src++;
    uint32_t w0_2 = *src++;
    uint32_t w0_3 = *src;
    uint16_t y, cb, cr;

    y = ((w0_0 >> 10U) & 0x3FFU) << 6U;
    cb = (w0_0 & 0x3FFU) << 6U;
    cr = ((w0_0 >> 20U) & 0x3FFU) << 6U;
    WRITE_RES(y, cb, cr);

    y = (w0_1 & 0x3FFU) << 6U;
    cb = (w0_0 & 0x3FFU) << 6U;
    cr = ((w0_0 >> 20U) & 0x3FFU) << 6U;
    WRITE_RES(y, cb, cr);

    y = ((w0_1 >> 20U) & 0x3FFU) << 6U;
    cb = ((w0_1 >> 10U) & 0x3FFU) << 6U;
    cr = (w0_2 & 0x3FFU) << 6U;
    WRITE_RES(y, cb, cr);

    y = ((w0_2 >> 10U) & 0x3FFU) << 6U;
    cb = ((w0_1 >> 10U) & 0x3FFU) << 6U;
    cr = (w0_2 & 0x3FFU) << 6U;
    WRITE_RES(y, cb, cr);

    y = (w0_3 & 0x3FFU) << 6U;
    cb = ((w0_2 >> 20U) & 0x3FFU) << 6U;
    cr = ((w0_3 >> 10U) & 0x3FFU) << 6U;
    WRITE_RES(y, cb, cr);

    y = ((w0_3 >> 20U) & 0x3FFU) << 6U;
    cb = ((w0_2 >> 20U) & 0x3FFU) << 6U;
    cr = ((w0_3 >> 10U) & 0x3FFU) << 6U;
    WRITE_RES(y, cb, cr);
}

template<typename IN_T, int bit_shift, bool has_alpha>
__global__ void convert_rgb_from_inter(int width, int height, int pitch_in, char *in)
{
    AVFrame *out_frame = &gpu_frame;
    size_t x = blockDim.x * blockIdx.x + threadIdx.x;
    size_t y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    void *dst_row = out_frame->data[0] + out_frame->linesize[0] *  y;
    void *src_row = in + y * pitch_in;

    IN_T *dst = ((IN_T *) dst_row) + (has_alpha ? 4 : 3) * x;
    uint16_t *src = ((uint16_t *) src_row) + 4 * x;

    *dst++ = *src++ >> bit_shift;
    *dst++ = *src++ >> bit_shift;
    *dst++ = *src++ >> bit_shift;
    if constexpr (has_alpha){
        *dst = *src >> bit_shift;
    }
}

template<bool has_alpha>
__global__ void convert_vuya_from_inter(int width, int height, int pitch_in, char *in)
{
    AVFrame *out_frame = &gpu_frame;
    size_t x = blockDim.x * blockIdx.x + threadIdx.x;
    size_t y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    void *dst_row = out_frame->data[0] + out_frame->linesize[0] * y;
    void *src_row = in + y * pitch_in;

    char *dst = ((char *) dst_row) + (has_alpha ? 4 : 3) * x;
    uint16_t *src = ((uint16_t *) src_row) + 4 * x;

    //uyva -> vuya
    *dst++ = src[2] >> 8U;
    *dst++ = src[0] >> 8U;
    *dst++ = src[1] >> 8U;
    if constexpr (has_alpha)
        *dst = src[3] >> 8U;
    else
        *dst = 0xFFFF;
}

template<typename OUT_T, int subsampling, int bit_shift>
__global__ void convert_yuvp_from_inter(int width, int height, int pitch_in, char *in){
    // yuv 444 i -> yuvp
    AVFrame *out_frame = &gpu_frame;
    size_t x = blockDim.x * blockIdx.x + threadIdx.x;
    size_t y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x >= width /2 || y >= height/2)
        return;

    OUT_T * dst_cb1, *dst_cb2, *dst_cr1, *dst_cr2, *dst_y1, *dst_y2;

    //y1, y2
    void * dst_y1_row = out_frame->data[0] + out_frame->linesize[0] * 2 * y;
    void * dst_y2_row = out_frame->data[0] + out_frame->linesize[0] * (2 * y + 1);
    dst_y1 = ((OUT_T *) (dst_y1_row)) + 2 * x;
    dst_y2 = ((OUT_T *) (dst_y2_row)) + 2 * x;

    //dst
    void * src1_row = in +  2 * y      * pitch_in;
    void * src2_row = in + (2 * y + 1) * pitch_in;
    uint16_t *src1 = ((uint16_t *) src1_row) + 4 * 2 * x;
    uint16_t *src2 = ((uint16_t *) src2_row) + 4 * 2 * x;

    //loops
    OUT_T *cb_loop[2];
    OUT_T *cr_loop[2];

    if constexpr (subsampling == 420){
        //fills the same cr/cb to each line
        void * dst_cb_row = out_frame->data[1] + out_frame->linesize[1] *  y;
        void * dst_cr_row = out_frame->data[2] + out_frame->linesize[2] *  y;

        dst_cb1 = ((OUT_T *) (dst_cb_row)) + x;
        dst_cr1 = ((OUT_T *) (dst_cr_row)) + x;
        cb_loop[0] = cb_loop[1] = dst_cb1;
        cr_loop[0] = cr_loop[1] = dst_cr1;

    } else if constexpr (subsampling == 422){
        //fills different cr/cb for each line
        void * dst_cb1_row = out_frame->data[1] + out_frame->linesize[1] *  2 * y;
        void * dst_cb2_row = out_frame->data[1] + out_frame->linesize[1] *  (2 * y + 1);

        void * dst_cr1_row = out_frame->data[2] + out_frame->linesize[2] *  2 * y;
        void * dst_cr2_row = out_frame->data[2] + out_frame->linesize[2] *  (2 * y + 1);

        dst_cb1 = ((OUT_T *) (dst_cb1_row)) + x;
        dst_cb2 = ((OUT_T *) (dst_cb2_row)) + x;

        dst_cr1 = ((OUT_T *) (dst_cr2_row)) + x;
        dst_cr2 = ((OUT_T *) (dst_cr1_row)) + x;

        cb_loop[0] = dst_cb1;
        cb_loop[1] = dst_cb2;

        cr_loop[0] = dst_cr1;
        cr_loop[1] = dst_cr2;
    }

    OUT_T *y_loop[2] = {dst_y1, dst_y2};
    uint16_t *src_loop[2] = {src1, src2};

    // each thread does 2 x 2 pixels
    for (int i = 0; i <2; ++i){
        uint16_t *src = src_loop[i];
        OUT_T *dst_y = y_loop[i];
        OUT_T *dst_cb = cb_loop[i];
        OUT_T *dst_cr = cr_loop[i];

        *dst_cb = ((src[0] + src[4]) / 2) >> bit_shift; // U
        *dst_y++ = src[1] >> bit_shift; // Y
        *dst_cr = ((src[2] + src[6]) / 2) >> bit_shift; // V
        *dst_y = src[5] >> bit_shift; // Y
    }
}

template<typename IN_T, int bit_shift>
__global__ void convert_yuv444p_from_inter(int width, int height, int pitch_in, char * in){
    //yuv444p -> yuv444i
    AVFrame *out_frame = &gpu_frame;
    size_t x = blockDim.x * blockIdx.x + threadIdx.x;
    size_t y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    void *dst_y1_row = out_frame->data[0] + out_frame->linesize[0] *  y;
    void *dst_cb_row = out_frame->data[1] + out_frame->linesize[1] *  y;
    void *dst_cr_row = out_frame->data[2] + out_frame->linesize[2] *  y;
    void *src_row = in + y * pitch_in;

    IN_T   *src_y1 = ((IN_T *) dst_y1_row) + x;
    IN_T   *src_cb = ((IN_T *) dst_cb_row) + x;
    IN_T   *src_cr = ((IN_T *) dst_cr_row) + x;
    uint16_t *dst = ((uint16_t *) src_row) + 4 * x;

    *src_cb = *dst++ >> bit_shift; // U
    *src_y1 = *dst++ >> bit_shift; // Y
    *src_cr = *dst >> bit_shift; // V
}


template<typename IN_T, int bit_shift>
__global__ void convert_p010le_from_inter(int width, int height, int pitch_in, char * in)
{
    // y cbcr -> yuv 444 i
    AVFrame *out_frame = &gpu_frame;
    size_t x = blockDim.x * blockIdx.x + threadIdx.x;
    size_t y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x >= width /2 || y >= height/2)
        return;

    IN_T * dst_cbcr, *dst_y1, *dst_y2;

    //y1, y2
    void * dst_y1_row = out_frame->data[0] + out_frame->linesize[0] * 2 * y;
    void * dst_y2_row = out_frame->data[0] + out_frame->linesize[0] * (2 * y + 1);
    dst_y1 = ((IN_T *) (dst_y1_row)) + 2 * x;
    dst_y2 = ((IN_T *) (dst_y2_row)) + 2 * x;

    void *dst_cbcr_row = out_frame->data[1] + out_frame->linesize[1] * y;
    dst_cbcr = ((IN_T *) dst_cbcr_row) + 2 * x;

    //src
    void *src1_row = in + (y * 2) * pitch_in;
    void *src2_row = in + (y * 2 +1) * pitch_in;
    uint16_t *src1 = ((uint16_t *) src1_row) + 4 * 2 * x;
    uint16_t *src2 = ((uint16_t *) src2_row) + 4 * 2 * x;

    // U
    *dst_cbcr++ = ((src1[0] + src2[0] + src1[4] + src2[4]) / 4) >> bit_shift;

    // Y
    *dst_y1++ = src1[1] >> bit_shift;
    *dst_y2++ = src2[1] >> bit_shift;

    // V
    *dst_cbcr = ((src1[2] + src2[2] + src1[6] + src2[6]) / 4) >> bit_shift;

    // Y
    *dst_y1 = src1[5] >> bit_shift;
    *dst_y2 = src2[5] >> bit_shift;
}


__global__ void convert_ayuv64_from_inter(int width, int height, int pitch_in, char * in)
{
    AVFrame *out_frame = &gpu_frame;
    size_t x = blockDim.x * blockIdx.x + threadIdx.x;
    size_t y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    void *dst_row = out_frame->data[0] + out_frame->linesize[0] *  y;
    void *src_row = in + y * pitch_in;

    uint16_t * src = ((uint16_t *) src_row) + 4 * x;
    uint16_t * dst = ((uint16_t *) dst_row) + 4 * x;

    *dst++ = src[3];
    *dst++ = src[1];
    *dst++ = src[0];
    *dst = src[2];
}


__global__ void convert_y210_from_inter(int width, int height, int pitch_in, char * in){
    AVFrame *out_frame = &gpu_frame;
    size_t x = blockDim.x * blockIdx.x + threadIdx.x;
    size_t y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    void *dst_row = out_frame->data[0] + out_frame->linesize[0] *  y;
    void *src_row = in + y * pitch_in;

    char *dst = ((char *) dst_row) + 4 * x;
    uint16_t *src = ((uint16_t *) src_row) + 4 * 2 * x;

    uint16_t y0, u, y1, v;
    //uyva
    y0 = src[1];
    u = (src[0] + src[4]) / 2;
    y1 = src[5];
    v = (src[2] + src[6]) / 2;

    *dst++ = y0 >> 8;
    *dst++ = u >> 8;
    *dst++ = y1 >> 8;
    *dst = v >> 8;
}

__global__ void convert_p210_from_inter(int width, int height, int pitch_in, char * in){
    AVFrame *out_frame = &gpu_frame;
    size_t x = blockDim.x * blockIdx.x + threadIdx.x;
    size_t y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x >= width / 2 || y >= height)
        return;

    void *dst_y_row = out_frame->data[0] + out_frame->linesize[0] * y;
    void *dst_cbcr_row = out_frame->data[1] + out_frame->linesize[1] * y;
    void *src_row = in + y * pitch_in;

    uint16_t *dst_cbcr = ((uint16_t *) dst_cbcr_row) + 2 * x;
    uint16_t *dst_y = ((uint16_t *) dst_y_row) + 2 * x;
    uint16_t *src = ((uint16_t *) src_row) + 2 * 4 * x;

    uint16_t cb, cr;
    //uyva -> y cbcr
    cb = (src[0] + src[4]) / 2;
    cr = (src[2] + src[6]) / 2;

    *dst_cbcr++ = cb;
    *dst_cbcr++ = cr;

    *dst_y++ = src[1];
    *dst_y++ = src[5];
}
/**************************************************************************************************************/
/*                                             KERNELS TO                                                     */
/**************************************************************************************************************/

template<typename IN_T, int bit_shift, bool has_alpha, codec_t CODEC>
__global__ void convert_rgb_to_yuv_inter(int width, int height, int pitch_in, int pitch_out, char * in, char *out){

    size_t x = blockDim.x * blockIdx.x + threadIdx.x;
    size_t y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    void *src_row = in + pitch_in * y;
    void *dst_row = out + y * pitch_out;

    IN_T *src = ((IN_T *) src_row) + (has_alpha ? 4 : 3) * x;
    uint16_t *dst = ((uint16_t *) dst_row) + 4 * x;

    comp_type_t r, g, b, y1, u, v;
    if constexpr (CODEC == R10k){
        uint8_t byte1 = *src++;
        uint8_t byte2 = *src++;
        uint8_t byte3 = *src++;
        uint8_t byte4 = *src++;

        r = byte1 << 8U | (byte2 & 0xC0U);
        g = (byte2 & 0x3FU) << 10U | (byte3 & 0xF0U) << 2U;
        b = (byte3 & 0xFU) << 12U | (byte4 & 0xFCU) << 4U;
    } else{
        r = *src++ << bit_shift;
        g = *src++ << bit_shift;
        b = *src++ << bit_shift;
    }

    u = (RGB_TO_CB_709_SCALED(r, g, b) >> COMP_BASE)+ (1<<15);
    y1 = (RGB_TO_Y_709_SCALED(r, g, b) >> COMP_BASE)+ (1<<12);
    v = (RGB_TO_CR_709_SCALED(r, g, b) >> COMP_BASE)+ (1<<15);

    *dst++ = CLAMP_LIMITED_CBCR(u, 16);
    *dst++ = CLAMP_LIMITED_Y(y1, 16);
    *dst++ = CLAMP_LIMITED_CBCR(v, 16);

    if constexpr (has_alpha){
        *dst = *src << bit_shift;
    } else{
        *dst = 0xFFFFU;
    }
}


template<typename IN_T, int bit_shift, bool has_alpha, codec_t CODEC>
__global__ void convert_rgb_to_rgb_inter(int width, int height, int pitch_in, int pitch_out, char * in, char *out){

    size_t x = blockDim.x * blockIdx.x + threadIdx.x;
    size_t y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    void *src_row = in + pitch_in * y;
    void *dst_row = out + y * pitch_out;

    IN_T *src = ((IN_T *) src_row) + (has_alpha ? 4 : 3) * x;
    uint16_t *dst = ((uint16_t *) dst_row) + 4 * x;

    comp_type_t r, g, b, y1, u, v;
    if constexpr (CODEC == R10k){
        uint8_t byte1 = *src++;
        uint8_t byte2 = *src++;
        uint8_t byte3 = *src++;
        uint8_t byte4 = *src++;

        r = byte1 << 8U | (byte2 & 0xC0U);
        g = (byte2 & 0x3FU) << 10U | (byte3 & 0xF0U) << 2U;
        b = (byte3 & 0xFU) << 12U | (byte4 & 0xFCU) << 4U;
    } else{
        r = *src++ << bit_shift;
        g = *src++ << bit_shift;
        b = *src++ << bit_shift;
    }

    *dst++ = r;
    *dst++ = g;
    *dst++ = b;

    if constexpr (has_alpha){
        *dst = *src << bit_shift;
    } else{
        *dst = 0xFFFFU;
    }
}

__global__ void convert_y416_to_rgb_inter(int width, int height, int pitch_in, int pitch_out, char * in, char *out){

    size_t x = blockDim.x * blockIdx.x + threadIdx.x;
    size_t y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    void *src_row = in + pitch_in * y;
    void *dst_row = out + y * pitch_out;

    uint16_t *src = ((uint16_t *) src_row) + 4 * x;
    uint16_t *dst = ((uint16_t *) dst_row) + 4 * x;

    comp_type_t r, g, b, y1, u, v;

    u = *src++;
    y1 = *src++;
    v = *src++;

    u = u - (1<<15);
    y1 = Y_SCALE * (y1 - (1<<12));
    v = v - (1<<15);

    r = YCBCR_TO_R_709_SCALED(y1, u, v) >> COMP_BASE;
    g = YCBCR_TO_G_709_SCALED(y1, u, v) >> COMP_BASE;
    b = YCBCR_TO_B_709_SCALED(y1, u, v) >> COMP_BASE;

    *dst++ = CLAMP_FULL(r, 16);
    *dst++ = CLAMP_FULL(g, 16);
    *dst++ = CLAMP_FULL(b, 16);

    *dst = *src;
}

template<typename IN_T, int bit_shift, bool is_reversed>
__global__ void convert_uyvy_to_yuv_inter(int width, int height, int pitch_in, int pitch_out, char * in, char *out){

    size_t x = blockDim.x * blockIdx.x + threadIdx.x;
    size_t y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x >= width / 2 || y >= height)
        return;

    void *src_row = in + pitch_in * y;
    void *dst_row = out + y * pitch_out;

    IN_T *src = ((IN_T *) src_row) + 4 * x;
    uint16_t *dst = ((uint16_t *) dst_row) + 4 * 2 * x;

    if constexpr (is_reversed){
        *dst++ =  src[1] << bit_shift; //U
        *dst++ =  src[0] << bit_shift; //Y1
        *dst++ =  src[3] << bit_shift; //V
        *dst++ =  0xFFFFU; //A

        *dst++ =  src[1] << bit_shift; //U
        *dst++ =  src[2] << bit_shift; //Y2
        *dst++ =  src[3] << bit_shift; //V
        *dst =  0xFFFFU; //A
    } else {
        *dst++ =  src[0] << bit_shift; //U
        *dst++ =  src[1] << bit_shift; //Y1
        *dst++ =  src[2] << bit_shift; //V
        *dst++ =  0xFFFFU; //A

        *dst++ =  src[0] << bit_shift; //U
        *dst++ =  src[3] << bit_shift; //Y2
        *dst++ =  src[2] << bit_shift; //V
        *dst =  0xFFFFU; //A
    }
}


template<typename IN_T, int bit_shift, bool is_reversed>
__global__ void convert_uyvy_to_rgb_inter(int width, int height, int pitch_in, int pitch_out, char * in, char *out){

    size_t x = blockDim.x * blockIdx.x + threadIdx.x;
    size_t y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x >= width / 2 || y >= height)
        return;

    void *src_row = in + pitch_in * y;
    void *dst_row = out + y * pitch_out;

    IN_T *src = ((IN_T *) src_row) + 4 * x;
    uint16_t *dst = ((uint16_t *) dst_row) + 4 * 2 * x;

    comp_type_t y1, y2, u, v, r ,g, b;

    if constexpr (is_reversed){
        y1 =  src[0] << bit_shift;
        u =  src[1] << bit_shift;
        y2 =  src[2] << bit_shift;
        v =  src[3] << bit_shift;
    } else {
        u =  src[0] << bit_shift;
        y1 =  src[1] << bit_shift;
        v =  src[2] << bit_shift;
        y2 =  src[3] << bit_shift;
    }
    u = u - (1<<15);
    y1 = Y_SCALE * (y1 - (1<<12));
    v = v - (1<<15);
    y2 = Y_SCALE * (y2 - (1<<12));

    r = YCBCR_TO_R_709_SCALED(y1, u, v) >> COMP_BASE;
    g = YCBCR_TO_G_709_SCALED(y1, u, v) >> COMP_BASE;
    b = YCBCR_TO_B_709_SCALED(y1, u, v) >> COMP_BASE;

    *dst++ = CLAMP_FULL(r, 16);
    *dst++ = CLAMP_FULL(g, 16);
    *dst++ = CLAMP_FULL(b, 16);
    *dst++ = 0xFFFFU;

    r = YCBCR_TO_R_709_SCALED(y2, u, v) >> COMP_BASE;
    g = YCBCR_TO_G_709_SCALED(y2, u, v) >> COMP_BASE;
    b = YCBCR_TO_B_709_SCALED(y2, u, v) >> COMP_BASE;

    *dst++ = CLAMP_FULL(r, 16);
    *dst++ = CLAMP_FULL(g, 16);
    *dst++ = CLAMP_FULL(b, 16);
    *dst = 0xFFFFU;
}


__global__ void convert_r12l_to_yuv_inter(int width, int height, int pitch_in, int pitch_out, char * in, char *out)
{
    size_t x = blockDim.x * blockIdx.x + threadIdx.x;
    size_t y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x >= width / 8 || y >= height)
        return;

    void *dst_row = out + pitch_out * y;
    uint16_t *dst = ((uint16_t *) dst_row) + 8 * 4 * x;

    const void * src_row = in + pitch_in * y;
    const uint8_t *src = ((uint8_t *) src_row) + 36 * x;

    auto WRITE_RES = [dst](comp_type_t &r, comp_type_t &g, comp_type_t &b) mutable {
        comp_type_t y1, u, v;
        y1 = (RGB_TO_Y_709_SCALED(r, g, b) >> COMP_BASE)+ (1<<12);
        u = (RGB_TO_CB_709_SCALED(r, g, b) >> COMP_BASE)+ (1<<15);
        v = (RGB_TO_CR_709_SCALED(r, g, b) >> COMP_BASE)+ (1<<15);

        *dst++ = CLAMP_LIMITED_CBCR(u, 16);
        *dst++ = CLAMP_LIMITED_Y(y1, 16);
        *dst++ = CLAMP_LIMITED_CBCR(v, 16);
        *dst++ = 0xFFFFU;
    };
    write_from_r12l(src, WRITE_RES);
}


__global__ void convert_r12l_to_rgb_inter(int width, int height, int pitch_in, int pitch_out, char * in, char *out)
{
    size_t x = blockDim.x * blockIdx.x + threadIdx.x;
    size_t y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x >= width / 8 || y >= height)
        return;

    void *dst_row = out + pitch_out * y;
    uint16_t *dst = ((uint16_t *) dst_row) + 8 * 4 * x;

    const void * src_row = in + pitch_in * y;
    const uint8_t *src = ((uint8_t *) src_row) + 36 * x;

    auto WRITE_RES = [dst](uint16_t r, uint16_t g, uint16_t b) mutable {
        *dst++ = r;
        *dst++ = g;
        *dst++ = b;
        *dst++ = 0xFFFFU;
    };
    write_from_r12l(src, WRITE_RES);
}

__global__ void convert_v210_to_rgb_inter(int width, int height, int pitch_in, int pitch_out, char * in, char *out)
{
    size_t x = blockDim.x * blockIdx.x + threadIdx.x;
    size_t y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x >= width / 6 || y >= height)
        return;

    void *dst_row = out + pitch_out * y;
    uint16_t *dst = ((uint16_t *) dst_row) + 6 * 4 * x;

    const void * src_row = in + pitch_in * y;
    const uint32_t *src = ((uint32_t *) src_row) + 4 * x;

    auto WRITE_RES = [dst](uint16_t &y11, uint16_t &cb1, uint16_t &cr1) mutable {
        comp_type_t y1, u, v, r, g, b, a;

        u = cb1 - (1<<15);
        y1 = Y_SCALE * (y11 - (1<<12));
        v = cr1 - (1<<15);

        r = YCBCR_TO_R_709_SCALED(y1, u, v) >> COMP_BASE;
        g = YCBCR_TO_G_709_SCALED(y1, u, v) >> COMP_BASE;
        b = YCBCR_TO_B_709_SCALED(y1, u, v) >> COMP_BASE;
        *dst++ = CLAMP_FULL(r, 16);
        *dst++ = CLAMP_FULL(g, 16);
        *dst++ = CLAMP_FULL(b, 16);
        *dst++ = 0xFFFFU;
    };

    write_from_v210(src, WRITE_RES);
}

__global__ void convert_v210_to_yuv_inter(int width, int height, int pitch_in, int pitch_out, char * in, char *out)
{
    size_t x = blockDim.x * blockIdx.x + threadIdx.x;
    size_t y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x >= width / 6 || y >= height)
        return;

    void *dst_row = out + pitch_out * y;
    uint16_t *dst = ((uint16_t *) dst_row) + 6 * 4 * x;

    const void * src_row = in + pitch_in * y;
    const uint32_t *src = ((uint32_t *) src_row) + 4 * x;

    auto WRITE_RES = [dst](uint16_t &y1, uint16_t &cb, uint16_t &cr) mutable {
        *dst++ = cb;
        *dst++ = y1;
        *dst++ = cr;
        *dst++ = 0xFFFFU;
    };

    write_from_v210(src, WRITE_RES);
}


/**************************************************************************************************************/
/*                                               RGB TO                                                       */
/**************************************************************************************************************/

template<typename T, int shift, bool alpha, codec_t CODEC>
void rgb_to_yuv_inter(int width, int height){

    size_t pitch_in = vc_get_linesize(width, CODEC);
    size_t pitch_out = vc_get_linesize(width, Y416);

    dim3 grid = dim3((width + BLOCK_SIZE - 1) / BLOCK_SIZE, (height + BLOCK_SIZE - 1) / BLOCK_SIZE );
    dim3 block = dim3(BLOCK_SIZE, BLOCK_SIZE);
    convert_rgb_to_yuv_inter<T, shift, alpha, CODEC><<<grid, block>>>(width, height, pitch_in, pitch_out, gpu_in_buffer, intermediate_to);
}

template<typename T, int shift, bool alpha, codec_t CODEC>
void rgb_to_rgb_inter(int width, int height){

    size_t pitch_in = vc_get_linesize(width, CODEC);
    size_t pitch_out = vc_get_linesize(width, Y416);

    dim3 grid = dim3((width + BLOCK_SIZE - 1) / BLOCK_SIZE, (height + BLOCK_SIZE - 1) / BLOCK_SIZE );
    dim3 block = dim3(BLOCK_SIZE, BLOCK_SIZE);
    convert_rgb_to_rgb_inter<T, shift, alpha, CODEC><<<grid, block>>>(width, height, pitch_in, pitch_out, gpu_in_buffer, intermediate_to);
}

void y416_to_rgb_inter(int width, int height){

    size_t pitch_out = vc_get_linesize(width, Y416);

    dim3 grid = dim3((width + BLOCK_SIZE - 1) / BLOCK_SIZE, (height + BLOCK_SIZE - 1) / BLOCK_SIZE );
    dim3 block = dim3(BLOCK_SIZE, BLOCK_SIZE);
    convert_y416_to_rgb_inter<<<grid, block>>>(width, height, pitch_out, pitch_out, gpu_in_buffer, intermediate_to);
}



template<typename T, int shift, bool is_reversed>
void uyvy_to_yuv_inter(int width, int height){

    size_t pitch_in = vc_get_linesize(width, shift == 8 ? UYVY : Y216);
    size_t pitch_out = vc_get_linesize(width, Y416);

    dim3 grid = dim3((width / 2 + BLOCK_SIZE - 1) / BLOCK_SIZE, (height + BLOCK_SIZE - 1) / BLOCK_SIZE );
    dim3 block = dim3(BLOCK_SIZE, BLOCK_SIZE);
    convert_uyvy_to_yuv_inter<T, shift, is_reversed><<<grid, block>>>(width, height, pitch_in, pitch_out, gpu_in_buffer, intermediate_to);
}

template<typename T, int shift, bool is_reversed>
void uyvy_to_rgb_inter(int width, int height){

    size_t pitch_in = vc_get_linesize(width, shift == 8 ? UYVY : Y216);
    size_t pitch_out = vc_get_linesize(width, Y416);

    dim3 grid = dim3((width / 2 + BLOCK_SIZE - 1) / BLOCK_SIZE, (height + BLOCK_SIZE - 1) / BLOCK_SIZE );
    dim3 block = dim3(BLOCK_SIZE, BLOCK_SIZE);
    convert_uyvy_to_rgb_inter<T, shift, is_reversed><<<grid, block>>>(width, height, pitch_in, pitch_out, gpu_in_buffer, intermediate_to);
}

void r12l_to_yuv(int width, int height){
    size_t pitch_in = vc_get_linesize(width, R12L);
    size_t pitch_out = vc_get_linesize(width, Y416);

    dim3 grid = dim3((width / 2 + BLOCK_SIZE - 1) / BLOCK_SIZE, (height + BLOCK_SIZE - 1) / BLOCK_SIZE );
    dim3 block = dim3(BLOCK_SIZE, BLOCK_SIZE);
    convert_r12l_to_yuv_inter<<<grid, block>>>(width, height, pitch_in, pitch_out, gpu_in_buffer, intermediate_to);
}

void r12l_to_rgb(int width, int height){
    size_t pitch_in = vc_get_linesize(width, R12L);
    size_t pitch_out = vc_get_linesize(width, Y416);

    dim3 grid = dim3((width / 2 + BLOCK_SIZE - 1) / BLOCK_SIZE, (height + BLOCK_SIZE - 1) / BLOCK_SIZE );
    dim3 block = dim3(BLOCK_SIZE, BLOCK_SIZE);
    convert_r12l_to_rgb_inter<<<grid, block>>>(width, height, pitch_in, pitch_out, gpu_in_buffer, intermediate_to);
}

void v210_to_rgb(int width, int height){
    size_t pitch_in = vc_get_linesize(width, v210);
    size_t pitch_out = vc_get_linesize(width, Y416);

    dim3 grid = dim3((width / 6 + BLOCK_SIZE - 1) / BLOCK_SIZE, (height + BLOCK_SIZE - 1) / BLOCK_SIZE );
    dim3 block = dim3(BLOCK_SIZE, BLOCK_SIZE);
    convert_v210_to_rgb_inter<<<grid, block>>>(width, height, pitch_in, pitch_out, gpu_in_buffer, intermediate_to);
}

void v210_to_yuv(int width, int height){
    size_t pitch_in = vc_get_linesize(width, v210);
    size_t pitch_out = vc_get_linesize(width, Y416);

    dim3 grid = dim3((width / 6 + BLOCK_SIZE - 1) / BLOCK_SIZE, (height + BLOCK_SIZE - 1) / BLOCK_SIZE );
    dim3 block = dim3(BLOCK_SIZE, BLOCK_SIZE);
    convert_v210_to_yuv_inter<<<grid, block>>>(width, height, pitch_in, pitch_out, gpu_in_buffer, intermediate_to);
}
/**************************************************************************************************************/
/*                                              RGB FROM                                                      */
/**************************************************************************************************************/

template<typename T, int bit_shift, bool alpha>
void rgbp_from_inter(int width, int height){
    size_t pitch_in = vc_get_linesize(width, Y416);
    dim3 grid = dim3((width + BLOCK_SIZE - 1) / BLOCK_SIZE, (height + BLOCK_SIZE - 1) / BLOCK_SIZE );
    dim3 block = dim3(BLOCK_SIZE, BLOCK_SIZE);
    convert_rgbp_from_inter<T, bit_shift, alpha><<<grid, block>>>(width, height, pitch_in, intermediate_to);
}

void ayuv64_from_inter(int width, int height){
    size_t pitch_in = vc_get_linesize(width, Y416);
    dim3 grid = dim3((width + BLOCK_SIZE - 1) / BLOCK_SIZE, (height + BLOCK_SIZE - 1) / BLOCK_SIZE );
    dim3 block = dim3(BLOCK_SIZE, BLOCK_SIZE);
    convert_ayuv64_from_inter<<<grid, block>>>(width, height, pitch_in, intermediate_to);
}

template<typename T, int bit_shift, bool alpha>
void rgb_from_inter(int width, int height){
    size_t pitch_in = vc_get_linesize(width, Y416);
    dim3 grid = dim3((width + BLOCK_SIZE - 1) / BLOCK_SIZE, (height + BLOCK_SIZE - 1) / BLOCK_SIZE );
    dim3 block = dim3(BLOCK_SIZE, BLOCK_SIZE);
    convert_rgb_from_inter<T, bit_shift, alpha><<<grid, block>>>(width, height, pitch_in, intermediate_to);
}

template<typename T, int bit_shift, int subsampling>
void yuvp_from_inter(int width, int height){
    size_t pitch_in = vc_get_linesize(width, Y416);

    assert(subsampling == 422 || subsampling == 420);

    //execute the conversion
    dim3 grid = dim3((width / 2 + BLOCK_SIZE - 1) / BLOCK_SIZE, (height / 2 + BLOCK_SIZE - 1) / BLOCK_SIZE );;
    dim3 block = dim3(BLOCK_SIZE, BLOCK_SIZE);

    convert_yuvp_from_inter<T, subsampling, bit_shift><<<grid, block>>>(width, height, pitch_in, intermediate_to);
}

template<typename T, int bit_shift>
void yuv444p_from_inter(int width, int height){
    size_t pitch_in = vc_get_linesize(width, Y416);

    //execute the conversion
    dim3 grid = dim3((width + BLOCK_SIZE - 1) / BLOCK_SIZE, (height + BLOCK_SIZE - 1) / BLOCK_SIZE );;
    dim3 block = dim3(BLOCK_SIZE, BLOCK_SIZE);

    convert_yuv444p_from_inter<T, bit_shift><<<grid, block>>>(width, height, pitch_in, intermediate_to);
}

template<typename T, int bit_shift>
void p010le_from_inter(int width, int height){
    size_t pitch_in = vc_get_linesize(width, Y416);

    //execute the conversion
    dim3 grid = dim3((width / 2 + BLOCK_SIZE - 1) / BLOCK_SIZE, (height / 2 + BLOCK_SIZE - 1) / BLOCK_SIZE );;
    dim3 block = dim3(BLOCK_SIZE, BLOCK_SIZE);

    convert_p010le_from_inter<T, bit_shift><<<grid, block>>>(width, height, pitch_in, intermediate_to);
}

template<bool b>
void vuya_form_inter(int width, int height){

    size_t pitch_in = vc_get_linesize(width, Y416);

    dim3 grid = dim3((width + BLOCK_SIZE - 1) / BLOCK_SIZE, (height + BLOCK_SIZE - 1) / BLOCK_SIZE );
    dim3 block = dim3(BLOCK_SIZE, BLOCK_SIZE);
    convert_vuya_from_inter<b><<<grid, block>>>(width, height, pitch_in, intermediate_to);
}

void y210_form_inter(int width, int height){

    size_t pitch_in = vc_get_linesize(width, Y416);

    dim3 grid = dim3((width + BLOCK_SIZE - 1) / BLOCK_SIZE, (height + BLOCK_SIZE - 1) / BLOCK_SIZE );
    dim3 block = dim3(BLOCK_SIZE, BLOCK_SIZE);
    convert_y210_from_inter<<<grid, block>>>(width, height, pitch_in, intermediate_to);
}


void p210_from_inter(int width, int height){

    size_t pitch_in = vc_get_linesize(width, Y416);

    dim3 grid = dim3((width + BLOCK_SIZE - 1) / BLOCK_SIZE, (height + BLOCK_SIZE - 1) / BLOCK_SIZE );
    dim3 block = dim3(BLOCK_SIZE, BLOCK_SIZE);
    convert_p210_from_inter<<<grid, block>>>(width, height, pitch_in, intermediate_to);
}
/**************************************************************************************************************/
/*                                                LISTS                                                       */
/**************************************************************************************************************/

std::map<codec_t, void (*)(int, int)> conversions_to_rgb_inter = {
        {RGBA, rgb_to_rgb_inter<uint8_t, 8, true, RGBA>},
        {RGB, rgb_to_rgb_inter<uint8_t, 8, false, RGB>},
        {RG48, rgb_to_rgb_inter<uint16_t, 0, false, RG48>},
        {R10k, rgb_to_rgb_inter<char, 6, true, R10k>},

        {UYVY, uyvy_to_rgb_inter<uint8_t, 8, false>},
        {YUYV, uyvy_to_rgb_inter<uint8_t, 8, true>},
        {R12L, r12l_to_rgb},
        {v210, v210_to_rgb},
        {Y216, uyvy_to_rgb_inter<uint16_t, 0, true>}, //sus also
        {Y416, y416_to_rgb_inter},
};

std::map<codec_t, void (*)(int, int)> conversions_to_yuv_inter = {
        {RGBA, rgb_to_yuv_inter<uint8_t, 8, true, RGBA>},
        {RGB, rgb_to_yuv_inter<uint8_t, 8, false, RGB>},
        {RG48, rgb_to_yuv_inter<uint16_t, 0, false, RG48>},
        {R10k, rgb_to_yuv_inter<char, 6, true, R10k>},

        {UYVY, uyvy_to_yuv_inter<uint8_t, 8, false>},
        {YUYV, uyvy_to_yuv_inter<uint8_t, 8, false>},
        {R12L, r12l_to_yuv},
        {v210, v210_to_yuv},
        {Y216, uyvy_to_yuv_inter<uint16_t, 0, true>}, //y216 yuv420p sus colour
        {Y416, rgb_to_rgb_inter<uint16_t, 0, true, Y416>},
};

const std::map<AVPixelFormat, std::tuple<int, void (*)(int, int)>> conversions_from_inter = {
        // 10-bit YUV
        {AV_PIX_FMT_YUV420P10LE, {YUV_INTER_TO, yuvp_from_inter<uint16_t, 6, 420>}},
        {AV_PIX_FMT_YUV444P10LE, {YUV_INTER_TO, yuv444p_from_inter<uint16_t, 6>}},
        {AV_PIX_FMT_YUV422P10LE, {YUV_INTER_TO, yuvp_from_inter<uint16_t, 6, 422>}},
        {AV_PIX_FMT_P010LE, {YUV_INTER_TO, p010le_from_inter<uint16_t, 0>}},

        // 8-bit YUV (NV12)
        {AV_PIX_FMT_NV12, {YUV_INTER_TO, p010le_from_inter<char, 8>}},

        {AV_PIX_FMT_YUV420P, {YUV_INTER_TO, yuvp_from_inter<char, 8, 420>}},
        {AV_PIX_FMT_YUV422P, {YUV_INTER_TO, yuvp_from_inter<char, 8, 422>}},
        {AV_PIX_FMT_YUV444P, {YUV_INTER_TO, yuv444p_from_inter<char, 8>}},

        {AV_PIX_FMT_YUVJ420P, {YUV_INTER_TO, yuvp_from_inter<char, 8, 420>}},
        {AV_PIX_FMT_YUVJ422P, {YUV_INTER_TO, yuvp_from_inter<char, 8, 422>}},
        {AV_PIX_FMT_YUVJ444P, {YUV_INTER_TO, yuv444p_from_inter<char, 8>}},
        // 12-bit YUV
        {AV_PIX_FMT_YUV420P12LE, {YUV_INTER_TO, yuvp_from_inter<uint16_t, 4, 420>}},
        {AV_PIX_FMT_YUV422P12LE, {YUV_INTER_TO, yuvp_from_inter<uint16_t, 4, 422>}},
        {AV_PIX_FMT_YUV444P12LE, {YUV_INTER_TO, yuv444p_from_inter<uint16_t, 4>}},
        // 16-bit YUV
        {AV_PIX_FMT_YUV420P16LE, {YUV_INTER_TO, yuvp_from_inter<uint16_t , 0, 420>}},
        {AV_PIX_FMT_YUV422P16LE, {YUV_INTER_TO, yuvp_from_inter<uint16_t , 0, 422>}},
        {AV_PIX_FMT_YUV444P16LE, {YUV_INTER_TO, yuv444p_from_inter<uint16_t , 0>}},

        {AV_PIX_FMT_AYUV64LE, {YUV_INTER_TO, ayuv64_from_inter}},

        //GBRP
        {AV_PIX_FMT_GBRP, {RGB_INTER_TO, rgbp_from_inter<uint8_t, 8, false>}},
        {AV_PIX_FMT_GBRAP, {RGB_INTER_TO, rgbp_from_inter<uint8_t, 8, true>}},

        {AV_PIX_FMT_GBRP12LE, {RGB_INTER_TO, rgbp_from_inter<uint16_t, 4, false>}},
        {AV_PIX_FMT_GBRP10LE, {RGB_INTER_TO, rgbp_from_inter<uint16_t, 6, false>}},
        {AV_PIX_FMT_GBRP16LE, {RGB_INTER_TO, rgbp_from_inter<uint16_t, 0, false>}},

        {AV_PIX_FMT_GBRAP12LE, {RGB_INTER_TO, rgbp_from_inter<uint16_t, 4, true>}},
        {AV_PIX_FMT_GBRAP10LE, {RGB_INTER_TO, rgbp_from_inter<uint16_t, 6, true>}},
        {AV_PIX_FMT_GBRAP16LE, {RGB_INTER_TO, rgbp_from_inter<uint16_t, 0, true>}},

        //RGB
        {AV_PIX_FMT_RGB24, {RGB_INTER_TO, rgb_from_inter<uint8_t, 8, false>}},
        {AV_PIX_FMT_RGB48LE, {RGB_INTER_TO, rgb_from_inter<uint16_t, 0, false>}},

        {AV_PIX_FMT_RGBA64LE, {RGB_INTER_TO, rgb_from_inter<uint16_t, 0, true>}},
        {AV_PIX_FMT_RGBA, {RGB_INTER_TO, rgb_from_inter<uint8_t, 8, true>}},

        {AV_PIX_FMT_Y210, {YUV_INTER_TO, y210_form_inter}},
#if P210_PRESENT
        {AV_PIX_FMT_P210LE, p210_from_inter},
#endif
#if XV3X_PRESENT
        {AV_PIX_FMT_XV30, y210_form_inter}, //idk how to test these
        {AV_PIX_FMT_Y212, y210_form_inter}, //idk how to test these
#endif
#if VUYX_PRESENT
        {AV_PIX_FMT_VUYA, vuya_from_inter<true>}, //idk how to test these
        {AV_PIX_FMT_VUYX, vuya_from_inter<false>} //idk how to test these
#endif
};

/**************************************************************************************************************/
/*                                              INTERFACE                                                     */
/**************************************************************************************************************/

bool convert_to_lavc(codec_t UG_codec, AVFrame * dst_frame, const char *src) {

    auto [inter, converter_from_inter] = conversions_from_inter.at(static_cast<AVPixelFormat>(dst_frame->format));

    //copy the image to gpu
    cudaMemcpy(gpu_in_buffer, src, vc_get_datalen(dst_frame->width, dst_frame->height, UG_codec), cudaMemcpyHostToDevice);
    //copy the destination to gpu
    cudaMemcpyToSymbol(gpu_frame, &(gpu_wrapper.frame), sizeof(AVFrame));

    if (inter == YUV_INTER_TO){
        auto converter_to = conversions_to_yuv_inter.at(UG_codec);
        converter_to(dst_frame->width, dst_frame->height);
    } else if (inter == RGB_INTER_TO){
        auto converter_to = conversions_to_rgb_inter.at(UG_codec);
        converter_to(dst_frame->width, dst_frame->height);
    } else {
        //error
    }

    converter_from_inter(dst_frame->width, dst_frame->height);

    //copy the converted image back to the host
    gpu_wrapper.copy_to_host(dst_frame);
//    av_image_fill_arrays()???
    return true;
}

bool to_lavc_init(AVPixelFormat AV_codec, codec_t UG_codec, int width, int height, AVFrame **dst){
    if (conversions_from_inter.find(AV_codec) == conversions_from_inter.end()
        || conversions_to_rgb_inter.find(UG_codec) == conversions_to_rgb_inter.end()){ //both should contain same keys
        std::cout << "[to_lavc_converter] conversion not supported\n";
        return false;
    }

    cudaMalloc(&intermediate_to, vc_get_datalen(width, height, Y416));
    cudaMalloc(&gpu_in_buffer, vc_get_datalen(width, height, UG_codec));

    *dst = av_frame_alloc();
    (*dst)->width = width;
    (*dst)->height = height;
    (*dst)->format = AV_codec;
    av_frame_get_buffer(*dst, 0);

//    host_wrapper.alloc(*dst);
    gpu_wrapper.alloc(*dst);

    return true;
}

void to_lavc_destroy(char *ptr){
    cudaFree(intermediate_to);
    cudaFree(gpu_in_buffer);
    gpu_wrapper.free_from_device();
}
