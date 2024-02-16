#ifndef CONV_UTILS
#define CONV_UTILS
#include <libavutil/pixfmt.h>
#include "../src/config_unix.h"
#include "../src/video_codec.h"

#include "../src/libavcodec/from_lavc_vid_conv.h"
#include "../src/libavcodec/to_lavc_vid_conv.h"
#include "../src/libavcodec/lavc_common.h"
#include "../src/video_codec.h"

#ifdef WORDS_BIGENDIAN
#define BYTE_SWAP(x) (3 - x)
#else
#define BYTE_SWAP(x) x
#endif

// RAII wrapper for AVPixelformat
struct AVF_GPU_wrapper{
    AVFrame frame;

    AVF_GPU_wrapper(){
        frame.data[0] = nullptr;
        frame.data[1] = nullptr;
        frame.data[2] = nullptr;
        frame.data[3] = nullptr;
    }

    void alloc(const AVFrame* new_frame){
        frame = *new_frame;
        cudaMalloc(&(frame.data[0]), frame.linesize[0] * frame.height);
        cudaMalloc(&(frame.data[1]), frame.linesize[1] * frame.height);
        cudaMalloc(&(frame.data[2]), frame.linesize[2] * frame.height);
        cudaMalloc(&(frame.data[3]), frame.linesize[3] * frame.height);
    }

    void copy_to_device(const AVFrame *new_frame){
        cudaMemcpy(frame.data[0], new_frame->data[0], frame.linesize[0] * frame.height, cudaMemcpyHostToDevice);
        if (frame.data[1]) cudaMemcpy(frame.data[1], new_frame->data[1], frame.linesize[1] * frame.height, cudaMemcpyHostToDevice);
        if (frame.data[2]) cudaMemcpy(frame.data[2], new_frame->data[2], frame.linesize[2] * frame.height, cudaMemcpyHostToDevice);
        if (frame.data[3]) cudaMemcpy(frame.data[3], new_frame->data[3], frame.linesize[3] * frame.height, cudaMemcpyHostToDevice);
    }

    void copy_to_host(const AVFrame *new_frame){
        cudaMemcpy( new_frame->data[0],frame.data[0], frame.linesize[0] * frame.height, cudaMemcpyDeviceToHost);
        if (frame.data[1]) cudaMemcpy( new_frame->data[1],frame.data[1], frame.linesize[1] * frame.height, cudaMemcpyDeviceToHost);
        if (frame.data[2]) cudaMemcpy( new_frame->data[2],frame.data[2], frame.linesize[2] * frame.height, cudaMemcpyDeviceToHost);
        if (frame.data[3]) cudaMemcpy( new_frame->data[3],frame.data[3], frame.linesize[3] * frame.height, cudaMemcpyDeviceToHost);
    }

    void free_from_device(){
        cudaFree(frame.data[0]);
        cudaFree(frame.data[1]);
        cudaFree(frame.data[2]);
        cudaFree(frame.data[3]);
    }
};

struct AVF_HOST_wrapper{
    AVFrame frame;

    AVF_HOST_wrapper(){
        frame.data[0] = nullptr;
        frame.data[1] = nullptr;
        frame.data[2] = nullptr;
        frame.data[3] = nullptr;
    }

    void alloc(const AVFrame* new_frame){
        frame = *new_frame;
        cudaMallocHost(&(frame.data[0]), frame.linesize[0] * frame.height);
        cudaMallocHost(&(frame.data[1]), frame.linesize[1] * frame.height);
        cudaMallocHost(&(frame.data[2]), frame.linesize[2] * frame.height);
        cudaMallocHost(&(frame.data[3]), frame.linesize[3] * frame.height);
    }

    void copy_to_host(const AVFrame *new_frame){
        cudaMemcpy(frame.data[0], new_frame->data[0], frame.linesize[0] * frame.height, cudaMemcpyDeviceToHost);
        if (frame.data[1]) cudaMemcpy(frame.data[1], new_frame->data[1], frame.linesize[1] * frame.height, cudaMemcpyDeviceToHost);
        if (frame.data[2]) cudaMemcpy(frame.data[2], new_frame->data[2], frame.linesize[2] * frame.height, cudaMemcpyDeviceToHost);
        if (frame.data[3]) cudaMemcpy(frame.data[3], new_frame->data[3], frame.linesize[3] * frame.height, cudaMemcpyDeviceToHost);
    }

    void free_from_device(){
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
