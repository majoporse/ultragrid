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

#define YUV_INTER 0
#define RGB_INTER 1
#define BLOCK_SIZE 32

char *intermediate;
char *gpu_out_buffer;

__global__ void write_from_yuv_to_rgb(char * __restrict dst_buf, const char *__restrict src,
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

__global__ void write_from_rgb_to_rgb(char * __restrict dst_buf, const char *__restrict src_buf,
                                    int pitch, size_t pitch_in, int width, int height){

    size_t x = blockDim.x * blockIdx.x + threadIdx.x;
    size_t y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x >= width || y >= height )
        return;

    const void *src_row = src_buf + pitch_in * y;
    const uint16_t *src = ((const uint16_t *) src_row) + 4 * x;

    void * dst_row = dst_buf + pitch * y;
    uint32_t *dst = ((uint32_t *) dst_row) + x;

    unsigned r = *src++ >> 6;
    unsigned g = *src++ >> 6;
    unsigned b = *src++ >> 6;
    // B5-B0 XX | G3-G0 B9-B6 | R1-R0 G9-G4 | R9-R2
    *dst++ = (b & 0x3FU) << 26U | 0x3000000U | (g & 0xFU) << 20U | (b >> 6U) << 16U | (r & 0x3U) << 14U | (g >> 4U) << 8U | r >> 2U;
}

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
    *dst++ = src[0]; // A
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
        tmp = *src_cbcr++ << bit_shift;
        *dst1++ = tmp;
        *dst2++ = tmp;
        // Y
        *dst1++ = *src_y1++ << bit_shift;
        *dst2++ = *src_y2++ << bit_shift;
        // V
        tmp = *src_cbcr++ << bit_shift;
        *dst1++ = tmp;
        *dst2++ = tmp;

        //A
        *dst1++ = 0xFFFFU;
        *dst2++ = 0xFFFFU;
    }
}

template<typename IN_T, int subsampling, int bit_shift>
__global__ void yuvp_to_intermediate(char * __restrict dst_buffer, int pitch, int width, int height){
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

    //loops
    IN_T *cb_loop[2];
    IN_T *cr_loop[2];

    if constexpr (subsampling == 420){
        //fills the same cr/cb to each line
        void * src_cb_row = in_frame->data[1] + in_frame->linesize[1] *  y;
        void * src_cr_row = in_frame->data[2] + in_frame->linesize[2] *  y;

        src_cb1 = ((IN_T *) (src_cb_row)) + x;
        src_cr1 = ((IN_T *) (src_cr_row)) + x;
        cb_loop[0] = cb_loop[1] = src_cb1;
        cr_loop[0] = cr_loop[1] = src_cr1;

    } else if constexpr (subsampling == 422){
        //fills different cr/cb for each line
        void * src_cb1_row = in_frame->data[1] + in_frame->linesize[1] *  2 * y;
        void * src_cb2_row = in_frame->data[1] + in_frame->linesize[1] *  (2 * y + 1);

        void * src_cr1_row = in_frame->data[2] + in_frame->linesize[2] *  2 * y;
        void * src_cr2_row = in_frame->data[2] + in_frame->linesize[2] *  (2 * y + 1);

        src_cb1 = ((IN_T *) (src_cb1_row)) + x;
        src_cb2 = ((IN_T *) (src_cb2_row)) + x;

        src_cr1 = ((IN_T *) (src_cr2_row)) + x;
        src_cr2 = ((IN_T *) (src_cr1_row)) + x;

        cb_loop[0] = src_cb1;
        cb_loop[1] = src_cb2;

        cr_loop[0] = src_cr1;
        cr_loop[1] = src_cr2;
    }

    IN_T *y_loop[2] = {src_y1, src_y2};
    uint16_t *dst_loop[2] = {dst1, dst2};

    // each thread does 2 x 2 pixels
    for (int i = 0; i <2; ++i){
        uint16_t *dst = dst_loop[i];
        IN_T *src_y = y_loop[i];
        IN_T *src_cb = cb_loop[i];
        IN_T *src_cr = cr_loop[i];

        for (int _ = 0; _ < 2; ++_) {
            *dst++ = *src_cb << bit_shift; // U
            *dst++ = *src_y++ << bit_shift; // Y
            *dst++ = *src_cr << bit_shift; // V
            *dst++ = 0xFFFFU; // A
        }
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

    IN_T * __restrict src_y1 = ((IN_T *) src_y1_row) + x;
    IN_T * __restrict src_cb = ((IN_T *) src_cb_row) + x;
    IN_T * __restrict src_cr = ((IN_T *) src_cr_row) + x;
    uint16_t *          dst1 = ((uint16_t *) dst_row) + 4 * x;

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
        *dst = *src_a;
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
/**************************************************************************************************************/
/*                                              RGB FROM                                                      */
/**************************************************************************************************************/

//template<int bit_shift>
void convert_from_rgb_inter_to_rgb(char * __restrict dst, const AVFrame *frame){
    size_t width = frame->width;
    size_t height = frame->height;
    size_t pitch = vc_get_linesize(width, R10k);

    assert((uintptr_t) gpu_out_buffer % 4 == 0);
    assert((uintptr_t) frame->linesize[0] % 2 == 0); // Y
    assert((uintptr_t) frame->linesize[1] % 2 == 0); // U
    assert((uintptr_t) frame->linesize[2] % 2 == 0); // V

    //execute the conversion
    dim3 grid2 = dim3((width + BLOCK_SIZE - 1) / BLOCK_SIZE, (height + BLOCK_SIZE - 1) / BLOCK_SIZE );
    dim3 block = dim3(BLOCK_SIZE, BLOCK_SIZE);

    //same linesize as Y416
    write_from_rgb_to_rgb<<<grid2, block>>>(gpu_out_buffer, intermediate, pitch, vc_get_linesize(width, Y416), width, height);
}

/**************************************************************************************************************/
/*                                              YUV FROM                                                      */
/**************************************************************************************************************/

template<int out_bit_depth>
void convert_from_yuv_inter_to_rgb(char * __restrict dst, const AVFrame *frame)
{
    size_t width = frame->width;
    size_t height = frame->height;
    size_t pitch = vc_get_linesize(width, R10k);

    assert((uintptr_t) gpu_out_buffer % 4 == 0);
    assert((uintptr_t) frame->linesize[0] % 2 == 0); // Y
    assert((uintptr_t) frame->linesize[1] % 2 == 0); // U
    assert((uintptr_t) frame->linesize[2] % 2 == 0); // V

    assert(out_bit_depth == 24 || out_bit_depth == 30 || out_bit_depth == 32);

    //execute the conversion
    dim3 grid2 = dim3((width + BLOCK_SIZE - 1) / BLOCK_SIZE, (height + BLOCK_SIZE - 1) / BLOCK_SIZE );
    dim3 block = dim3(BLOCK_SIZE, BLOCK_SIZE);

    write_from_yuv_to_rgb<<<grid2, block>>>(gpu_out_buffer, intermediate, pitch, vc_get_linesize(width, Y416), width, height);
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
    dim3 grid2 = dim3((width + BLOCK_SIZE - 1) / BLOCK_SIZE, (height + BLOCK_SIZE - 1) / BLOCK_SIZE );
    dim3 block = dim3(BLOCK_SIZE, BLOCK_SIZE);

    p010le_to_inter<T, i><<<grid2, block>>>(intermediate, pitch, width, height);
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

template<typename T, int bit_shift, int subsampling>
int convert_yuvp_to_inter(const AVFrame *frame){
    size_t width = frame->width;
    size_t height = frame->height;
    size_t pitch = vc_get_linesize(width, Y416);

    assert((uintptr_t) gpu_out_buffer % 4 == 0);
    assert((uintptr_t) frame->linesize[0] % 2 == 0); // Y
    assert((uintptr_t) frame->linesize[1] % 2 == 0); // U
    assert((uintptr_t) frame->linesize[2] % 2 == 0); // V

    assert(subsampling == 422 || subsampling == 420 || subsampling == 444);

    //execute the conversion
    dim3 grid1 = dim3((width / 2 + BLOCK_SIZE - 1) / BLOCK_SIZE, (height / 2 + BLOCK_SIZE - 1) / BLOCK_SIZE );
    dim3 grid2 = dim3((width + BLOCK_SIZE - 1) / BLOCK_SIZE, (height + BLOCK_SIZE - 1) / BLOCK_SIZE );
    dim3 block = dim3(BLOCK_SIZE, BLOCK_SIZE);

    if constexpr (subsampling == 420){
        yuvp_to_intermediate<T, 420, bit_shift><<<grid1, block>>>(intermediate, pitch, width, height);
    } else if constexpr (subsampling == 422){
        yuvp_to_intermediate<T, 422, bit_shift><<<grid1, block>>>(intermediate, pitch, width, height);
    } else if constexpr (subsampling == 444){
        yuv444p_to_intermediate<T, bit_shift><<<grid2, block>>>(intermediate, pitch, width, height);
    }

    return YUV_INTER;
}

/**************************************************************************************************************/
/*                                               RGB TO                                                      */
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


const std::map<int, int (*) (const AVFrame *)> conversions_to_inter = {
        // 10-bit YUV
        {AV_PIX_FMT_YUV420P10LE, convert_yuvp_to_inter<uint16_t, 6, 420>},
        {AV_PIX_FMT_YUV444P10LE, convert_yuvp_to_inter<uint16_t, 6, 444>},
        {AV_PIX_FMT_YUV422P10LE, convert_yuvp_to_inter<uint16_t, 6, 422>},
        {AV_PIX_FMT_P010LE, convert_p010le_to_inter<uint16_t, 0>},

        // 8-bit YUV (NV12)
        {AV_PIX_FMT_NV12, convert_p010le_to_inter<char, 8>},

        {AV_PIX_FMT_YUV420P, convert_yuvp_to_inter<char, 8, 420>},
        {AV_PIX_FMT_YUV422P, convert_yuvp_to_inter<char, 8, 422>},
        {AV_PIX_FMT_YUV444P, convert_yuvp_to_inter<char, 8, 444>},

        {AV_PIX_FMT_YUVJ420P, convert_yuvp_to_inter<char, 8, 420>},
        {AV_PIX_FMT_YUVJ422P, convert_yuvp_to_inter<char, 8, 422>},
        {AV_PIX_FMT_YUVJ444P, convert_yuvp_to_inter<char, 8, 444>},
        // 12-bit YUV
        {AV_PIX_FMT_YUV420P12LE, convert_yuvp_to_inter<uint16_t, 4, 420>},
        {AV_PIX_FMT_YUV422P12LE, convert_yuvp_to_inter<uint16_t, 4, 422>},
        {AV_PIX_FMT_YUV444P12LE, convert_yuvp_to_inter<uint16_t, 4, 444>},
         // 16-bit YUV
        {AV_PIX_FMT_YUV420P16LE, convert_yuvp_to_inter<uint16_t , 0, 420>},
        {AV_PIX_FMT_YUV422P16LE, convert_yuvp_to_inter<uint16_t , 0, 422>},
        {AV_PIX_FMT_YUV444P16LE, convert_yuvp_to_inter<uint16_t , 0, 444>},

        {AV_PIX_FMT_AYUV64, convert_ayuv64_to_y416},

        //GBR
        {AV_PIX_FMT_GBRP, convert_grb_to_inter<char, 8, false>},
        {AV_PIX_FMT_GBRAP, convert_grb_to_inter<char, 8, true>},

        {AV_PIX_FMT_GBRP10LE, convert_grb_to_inter<uint16_t, 6, false>},
        {AV_PIX_FMT_GBRP12LE, convert_grb_to_inter<uint16_t, 4, false>},
        {AV_PIX_FMT_GBRP16LE, convert_grb_to_inter<uint16_t, 0, false>},

        {AV_PIX_FMT_GBRAP10LE, convert_grb_to_inter<uint16_t, 6, true>},
        {AV_PIX_FMT_GBRAP12LE, convert_grb_to_inter<uint16_t, 4, true>},
        {AV_PIX_FMT_GBRAP16LE, convert_grb_to_inter<uint16_t, 0, true>},
        //RGB
        {AV_PIX_FMT_RGB24, convert_rgb_to_inter<char, 8, false>},
        {AV_PIX_FMT_RGB48LE, convert_rgb_to_inter<uint16_t, 0, false>},

        {AV_PIX_FMT_RGBA64LE, convert_rgb_to_inter<uint16_t, 0, true>},
        {AV_PIX_FMT_RGBA, convert_rgb_to_inter<char, 8, true>},

};

const std::map<int, void (*) (char * __restrict dst, const AVFrame *frame)> conversions_from_yuv_inter = {
        {R10k, convert_from_yuv_inter_to_rgb<30>}
};

const std::map<int, void (*) (char *, const AVFrame *)> conversions_from_rgb_inter = {
        {R10k, convert_from_rgb_inter_to_rgb}
};

bool convert_from_lavc( const AVFrame* frame, char *dst, codec_t to) {

    //RAII wrapper for gpu allocations
    AVF_GPU_wrapper new_frame(frame);

    //copy host avframe to device
    cudaMemcpyToSymbol(gpu_frame, &(new_frame.frame), sizeof(AVFrame));
//    std::cout << frame->format << '\n';

    auto converter_to = conversions_to_inter.at(frame->format);
    auto format = converter_to(frame);

    if (format == YUV_INTER){
        auto converter_from = conversions_from_yuv_inter.at(to);
        converter_from(dst, frame);
    } else if (format == RGB_INTER){
        auto converter_from = conversions_from_rgb_inter.at(to);
        converter_from(dst, frame);
    } else {
        //error
    }

    //copy the converted image back to the host
    cudaMemcpy(dst, gpu_out_buffer, vc_get_datalen(frame->width, frame->height, to), cudaMemcpyDeviceToHost);
    return true;
}

bool from_lavc_init(const AVFrame* frame, codec_t out, char **dst_ptr){
    if (conversions_to_inter.find(frame->format) == conversions_to_inter.end()
        || conversions_from_yuv_inter.find(out) == conversions_from_yuv_inter.end()){
        std::cout << "conversion not supported";
        return false;
    }
    cudaMalloc(&intermediate, vc_get_datalen(frame->width, frame->height, Y416));
    cudaMalloc(&gpu_out_buffer, vc_get_datalen(frame->width, frame->height, out));
    cudaMallocHost(dst_ptr, vc_get_datalen(frame->width, frame->height, out));
    return true;
}

void from_lavc_destroy(char *ptr){
    cudaFreeHost(ptr);
    cudaFree(intermediate);
    cudaFree(gpu_out_buffer);
}
