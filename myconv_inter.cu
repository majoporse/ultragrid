#include "myconv_inter.h"
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

char *intermediate;
char *gpu_out_buffer;

__global__ void write_form_intermediate(char * __restrict dst_buf, const char *__restrict src,
                int pitch, size_t pitch_in, int width, int height){

    // yuv 4:4:4 interleaved -> R10k
    size_t x_pos = blockDim.x * blockIdx.x + threadIdx.x;
    size_t y_pos = blockDim.y * blockIdx.y + threadIdx.y;

    if (x_pos >= width || y_pos >= height )
        return;

    char *dst = dst_buf + y_pos * pitch + 4 * x_pos;

//    assert((uintptr_t) src % 2 == 0);
    const uint16_t *in = (uint16_t *) (src + y_pos * pitch_in + 4 * 2 * x_pos) ;
    comp_type_t y, u, v, r, g, b;

    u = *in++ - (1<<15);
    y = Y_SCALE * (*in++ - (1<<12));
    v = *in - (1<<15);

    r = (YCBCR_TO_R_709_SCALED(y, u, v) >> (COMP_BASE + 6U));
    g = (YCBCR_TO_G_709_SCALED(y, u, v) >> (COMP_BASE + 6U));
    b = (YCBCR_TO_B_709_SCALED(y, u, v) >> (COMP_BASE + 6U));
    r = CLAMP_FULL(r, 10);
    g = CLAMP_FULL(g, 10);
    b = CLAMP_FULL(b, 10);

    uint32_t res;
    char *res_ptr = (char *) &res;

    *res_ptr++ = r >> 2U;
    *res_ptr++ = (r & 0x3U) << 6U | g >> 4U;
    *res_ptr++ = (g & 0xFU) << 4U | b >> 6U;
    *res_ptr++ = (b & 0x3FU) << 2U;

    *(uint32_t *) dst = res;
}

__global__ void write_to_intermediate(char * __restrict dst_buffer, int pitch, int width, int height){
    //yuv 4:2:0 planar -> yuv 4:4:4 interleaved

    AVFrame *in_frame = &gpu_frame;
    size_t x = blockDim.x * blockIdx.x + threadIdx.x;
    size_t y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x >= width /2 || y >= height/2)
        return;

    uint16_t * __restrict src_y1 = (uint16_t *)(void *)(in_frame->data[0] + in_frame->linesize[0] * 2 * y        + 4 * x);
    uint16_t * __restrict src_y2 = (uint16_t *)(void *)(in_frame->data[0] + in_frame->linesize[0] * ( 2 * y + 1) + 4 * x);
    uint16_t * __restrict src_cb = (uint16_t *)(void *)(in_frame->data[1] + in_frame->linesize[1] *  y        + 2 * x);
    uint16_t * __restrict src_cr = (uint16_t *)(void *)(in_frame->data[2] + in_frame->linesize[2] *  y        + 2 * x);

    uint16_t *dst1 = (uint16_t *)(dst_buffer + 2 * y       * pitch) + 4 * 2 * x;
    uint16_t *dst2 = (uint16_t *)(dst_buffer + (2 * y + 1) * pitch) + 4 * 2 * x;
    // each thread does 2 x 2 pixels
    for (int _ = 0; _ < 2; ++_) {
        uint64_t res = *(uint64_t *) dst1;
        uint16_t *res_ptr = (uint16_t *) &res;
        *res_ptr++ = *src_cb << (16U - 10); // U
        *res_ptr++ = *src_y1++ << (16U - 10); // Y
        *res_ptr++ = *src_cr << (16U - 10); // V
        *res_ptr++ = 0xFFFFU; // A
        *(uint64_t *) dst1 = res;
        dst1+= 4;
    }

    for (int _ = 0; _ < 2; ++_) {
        uint64_t res = *(uint64_t *) dst2;
        uint16_t *res_ptr = (uint16_t *) &res;
        *res_ptr++ = *src_cb << (16U - 10); // U
        *res_ptr++ = *src_y2++ << (16U - 10); // Y
        *res_ptr++ = *src_cr << (16U - 10); // V
        *res_ptr++ = 0xFFFFU; // A
        *(uint64_t *) dst2 = res;
        dst2+= 4;
    }
}


#define BLOCK_SIZE 32

void convert_from_lavc_yuv_to_rgb(int subsampling, char * __restrict dst, const AVFrame *frame, int out_bit_depth)
{
    size_t width = frame->width;
    size_t height = frame->height;
    size_t pitch = vc_get_linesize(width, R10k);

    assert((uintptr_t) gpu_out_buffer % 4 == 0);
    assert((uintptr_t) frame->linesize[0] % 2 == 0); // Y
    assert((uintptr_t) frame->linesize[1] % 2 == 0); // U
    assert((uintptr_t) frame->linesize[2] % 2 == 0); // V


    assert(subsampling == 422 || subsampling == 420);
    assert(out_bit_depth == 24 || out_bit_depth == 30 || out_bit_depth == 32);

    //RAII wrapper for gpu allocations
    AVF_GPU_wrapper new_frame(frame);

    //copy host avframe to device
    cudaMemcpyToSymbol(gpu_frame, &(new_frame.frame), sizeof(AVFrame));

    //execute the conversion
    dim3 grid1 = dim3((width / 2 + BLOCK_SIZE - 1) / BLOCK_SIZE, (height / 2 + BLOCK_SIZE - 1) / BLOCK_SIZE );
    dim3 grid2 = dim3((width + BLOCK_SIZE - 1) / BLOCK_SIZE, (height + BLOCK_SIZE - 1) / BLOCK_SIZE );
    dim3 block = dim3(BLOCK_SIZE, BLOCK_SIZE);

    write_to_intermediate<<<grid1, block>>>(intermediate, vc_get_linesize(width, Y416), width, height);
    write_form_intermediate<<<grid2, block>>>(gpu_out_buffer, intermediate, pitch, vc_get_linesize(width, Y416), width, height);

    //copy the converted image back to the host
    cudaMemcpy(dst, gpu_out_buffer, vc_get_datalen(width, height, R10k), cudaMemcpyDeviceToHost);
}

template<AVPixelFormat AV_PIX_FMT_YUV420P10LE, codec_t R10k>
void convert_yuv_to_rgb(char * __restrict dst_buffer, const AVFrame *frame){
    convert_from_lavc_yuv_to_rgb(420, dst_buffer, frame, 30);
}

const std::map<std::tuple<AVPixelFormat, codec_t>, void (*) (char *, const AVFrame *)> conversions = {
        {{AV_PIX_FMT_YUV420P10LE, R10k}, convert_yuv_to_rgb<AV_PIX_FMT_YUV420P10LE, R10k>}
};

const std::map<int, codec_t> intermediate_t{
        {AV_PIX_FMT_YUV420P10LE, Y416}
};

conv_t get_conversion_from_lavc(AVPixelFormat from, codec_t to) { return conversions.at({from, to}); }

bool from_lavc_init(const AVFrame* frame, codec_t out){
    if (intermediate_t.find(frame->format) == intermediate_t.end()){
        std::cout << "conversion not supported";
        return false;
    }
    cudaMalloc(&intermediate, vc_get_datalen(frame->width, frame->height, intermediate_t.at(frame->format)));
    cudaMalloc(&gpu_out_buffer, vc_get_datalen(frame->width, frame->height, R10k));
    return true;
}

void from_lavc_destroy(){
    cudaFree(intermediate);
    cudaFree(gpu_out_buffer);
}
