#ifndef CONV_UTILS
#define CONV_UTILS
#include <libavutil/pixfmt.h>
#include "../src/config_unix.h"
#include "../src/video_codec.h"

#include "../src/libavcodec/from_lavc_vid_conv.h"
#include "../src/libavcodec/to_lavc_vid_conv.h"
#include "../src/libavcodec/lavc_common.h"
#include "../src/video_codec.h"

// RAII wrapper for AVPixelformat
struct AVF_GPU_wrapper{
    AVFrame frame;

    AVF_GPU_wrapper(const AVFrame* new_frame){
        frame = *new_frame;
        cudaMalloc(&(frame.data[0]), frame.linesize[0] * frame.height);
        cudaMalloc(&(frame.data[1]), frame.linesize[1] * frame.height);
        cudaMalloc(&(frame.data[2]), frame.linesize[2] * frame.height);
        cudaMalloc(&(frame.data[3]), frame.linesize[2] * frame.height);


        cudaMemcpy(frame.data[0], new_frame->data[0], frame.linesize[0] * frame.height, cudaMemcpyHostToDevice);
        if (frame.data[1]) cudaMemcpy(frame.data[1], new_frame->data[1], frame.linesize[1] * frame.height, cudaMemcpyHostToDevice);
        if (frame.data[2]) cudaMemcpy(frame.data[2], new_frame->data[2], frame.linesize[2] * frame.height, cudaMemcpyHostToDevice);
        if (frame.data[3]) cudaMemcpy(frame.data[2], new_frame->data[2], frame.linesize[2] * frame.height, cudaMemcpyHostToDevice);
    }

    ~AVF_GPU_wrapper(){
        cudaFree(frame.data[0]);
        cudaFree(frame.data[1]);
        cudaFree(frame.data[2]);
        cudaFree(frame.data[3]);
    }
};

//RAII wrapper for cuda allocations
struct cuda_alloc_wrapper{
    char *ptr = nullptr;
    cuda_alloc_wrapper(size_t size) { cudaMalloc(&ptr, size); }
    cuda_alloc_wrapper(){}
    ~cuda_alloc_wrapper() { cudaFree(ptr); }
};

#endif
