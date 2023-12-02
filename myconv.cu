#include "myconv.h"

#include <vector>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <iomanip>
#include "../src/color.h"
#include <stdlib.h>


#define R R_SHIFT_IDX
#define G G_SHIFT_IDX
#define B B_SHIFT_IDX

__device__ AVFrame gpu_frame;

template<int out_bit_depth, int subsampling>
__global__ void write(char * __restrict dst_buffer, int pitch,
                      const int * __restrict rgb_shift, int width, int height){

    AVFrame *frame = &gpu_frame;
    uint32_t alpha_mask = 0xFFFFFFFFU ^ (0xFFU << rgb_shift[R]) ^ (0xFFU << rgb_shift[G]) ^ (0xFFU << rgb_shift[B]);
    const int bpp = out_bit_depth == 30 ? 10 : 8;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;

    if (x * 2 >= width || y * 2 >= height)
        return;

    //y
    uint16_t * __restrict src_y1 = (uint16_t *)(void *) (frame->data[0] + frame->linesize[0] * 2 * y       + 4 * x);
    uint16_t * __restrict src_y2 = (uint16_t *)(void *) (frame->data[0] + frame->linesize[0] * (2 * y + 1) + 4 * x);
    uint16_t * __restrict src_cb1;
    uint16_t * __restrict src_cr1;
    uint16_t * __restrict src_cb2;
    uint16_t * __restrict src_cr2;

    if constexpr (subsampling == 420) {
        src_cb1 = src_cb2 = (uint16_t *)(void *) (frame->data[1] + frame->linesize[1] * y + 2 * x );
        src_cr1 = src_cr2 = (uint16_t *)(void *) (frame->data[2] + frame->linesize[2] * y + 2 * x );
    } else {
        src_cb1 = (uint16_t *)(void *) (frame->data[1] + frame->linesize[1] * (2 * y)     + 2 * x );
        src_cb2 = (uint16_t *)(void *) (frame->data[1] + frame->linesize[1] * (2 * y + 1) + 2 * x );
        src_cr1 = (uint16_t *)(void *) (frame->data[2] + frame->linesize[2] * (2 * y)     + 2 * x );
        src_cr2 = (uint16_t *)(void *) (frame->data[2] + frame->linesize[2] * (2 * y + 1) + 2 * x );
    }

    unsigned char *dst1 = (unsigned char *) dst_buffer + (2 * y)     * pitch  + 2 * x * (out_bit_depth == 24 ? 3 : 4);
    unsigned char *dst2 = (unsigned char *) dst_buffer + (2 * y + 1) * pitch  + 2 * x * (out_bit_depth == 24 ? 3 : 4);
    //x
    comp_type_t cr = *src_cr1++ - (1 << 9);
    comp_type_t cb = *src_cb1++ - (1 << 9);
    comp_type_t rr = YCBCR_TO_R_709_SCALED(0, cb, cr) >> (COMP_BASE + (10 - bpp));
    comp_type_t gg = YCBCR_TO_G_709_SCALED(0, cb, cr) >> (COMP_BASE + (10 - bpp));
    comp_type_t bb = YCBCR_TO_B_709_SCALED(0, cb, cr) >> (COMP_BASE + (10 - bpp));

    auto WRITE_RES_YUV10P_TO_RGB =
            [&](comp_type_t Y, unsigned char *&DST) {
                comp_type_t b = Y + bb;
                comp_type_t r = Y + rr;
                comp_type_t g = Y + gg;
                r = CLAMP_FULL(r, bpp);
                g = CLAMP_FULL(g, bpp);
                b = CLAMP_FULL(b, bpp);
                if constexpr (out_bit_depth == 32) {
                    // printf("0\n");
                    //    *((uint32_t *)(void *) (DST)) = alpha_mask | (r << rgb_shift[R] | g << rgb_shift[G] | b << rgb_shift[B]);
                    DST += 4;
                } else if constexpr (out_bit_depth == 24) {
                    // printf("1\n");
                    *DST++ = r;
                    *DST++ = g;
                    *DST++ = b;
                } else {
                    // printf("2\n");
                    *((uint32_t *) (void *) (DST)) =
                            r >> 2U | (r & 0x3U) << 14 | g >> 4U << 8U | (g & 0xFU) << 20U | b >> 6U << 16U |
                            (b & 0x3FU) << 26U | 0x3U << 24U;
                    /*      == htonl(r << 22U | g << 12U | b << 2U) */
                    DST += 4;
                }
            };


    comp_type_t y1 = (Y_SCALE * (*src_y1++ - (1 << 6))) >> (COMP_BASE + (10 - bpp));
    WRITE_RES_YUV10P_TO_RGB(y1, dst1);

    comp_type_t y11 = (Y_SCALE * (*src_y1 - (1 << 6))) >> (COMP_BASE + (10 - bpp));
    WRITE_RES_YUV10P_TO_RGB(y11, dst1);

    if constexpr (subsampling == 422) {
        cr = *src_cr2++ - (1 << 9);
        cb = *src_cb2++ - (1 << 9);
        rr = YCBCR_TO_R_709_SCALED(0, cb, cr) >> (COMP_BASE + (10 - bpp));
        gg = YCBCR_TO_G_709_SCALED(0, cb, cr) >> (COMP_BASE + (10 - bpp));
        bb = YCBCR_TO_B_709_SCALED(0, cb, cr) >> (COMP_BASE + (10 - bpp));
    }

    comp_type_t y2 = (Y_SCALE * (*src_y2++ - (1 << 6))) >> (COMP_BASE + (10 - bpp));
    WRITE_RES_YUV10P_TO_RGB(y2, dst2);

    comp_type_t y22 = (Y_SCALE * (*src_y2++ - (1 << 6))) >> (COMP_BASE + (10 - bpp));
    WRITE_RES_YUV10P_TO_RGB(y22, dst2);
}

struct AVF_GPU_wrapper{
    AVFrame frame;

    AVF_GPU_wrapper(AVFrame* new_frame){
        frame = *new_frame;
        cudaMalloc(&(frame.data[0]), frame.linesize[0] * frame.height);
        cudaMalloc(&(frame.data[1]), frame.linesize[1] * frame.height);
        cudaMalloc(&(frame.data[2]), frame.linesize[2] * frame.height);

        cudaMemcpy(frame.data[0], new_frame->data[0], frame.linesize[0] * frame.height, cudaMemcpyHostToDevice);
        cudaMemcpy(frame.data[1], new_frame->data[1], frame.linesize[1] * frame.height, cudaMemcpyHostToDevice);
        cudaMemcpy(frame.data[2], new_frame->data[2], frame.linesize[2] * frame.height, cudaMemcpyHostToDevice);
    }

    ~AVF_GPU_wrapper(){
        cudaFree(frame.data[0]);
        cudaFree(frame.data[1]);
        cudaFree(frame.data[2]);
    }
};

#define BLOCK_SIZE 32


void yuvp10le_to_rgb(int subsampling, char * __restrict dst_buffer, AVFrame *frame,
                     int width, int height, int pitch, const int * __restrict rgb_shift, int out_bit_depth)
{
    assert((uintptr_t) dst_buffer % 4 == 0);
    assert((uintptr_t) frame->linesize[0] % 2 == 0); // Y
    assert((uintptr_t) frame->linesize[1] % 2 == 0); // U
    assert((uintptr_t) frame->linesize[2] % 2 == 0); // V

    assert(subsampling == 422 || subsampling == 420);
    assert(out_bit_depth == 24 || out_bit_depth == 30 || out_bit_depth == 32);

    //RAII wrapper for gpu allocations
    AVF_GPU_wrapper new_frame(frame);

    //copy host avframe to device
    cudaMemcpyToSymbol(gpu_frame, &(new_frame.frame), sizeof(AVFrame));

    write<30, 420><<<dim3((width / 2 + BLOCK_SIZE - 1) / BLOCK_SIZE, (height / 2 + BLOCK_SIZE - 1) / BLOCK_SIZE ),
    dim3(BLOCK_SIZE, BLOCK_SIZE)>>>(dst_buffer, pitch,  rgb_shift, width, height);
}
