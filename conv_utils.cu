#include "conv_utils.h"
#include <iostream>

AVF_GPU_wrapper::AVF_GPU_wrapper(){
    frame.data[0] = nullptr;
    frame.data[1] = nullptr;
    frame.data[2] = nullptr;
    frame.data[3] = nullptr;
}

void AVF_GPU_wrapper::alloc(const AVFrame* new_frame) {
    frame = *new_frame;
    if (new_frame->format == AV_PIX_FMT_YUV420P ||
        new_frame->format == AV_PIX_FMT_YUV420P10LE ||
        new_frame->format == AV_PIX_FMT_YUV420P12LE ||
        new_frame->format == AV_PIX_FMT_YUV420P16LE ||
        new_frame->format == AV_PIX_FMT_NV12 ||
        new_frame->format == AV_PIX_FMT_YUVJ420P ||
        new_frame->format == AV_PIX_FMT_P010LE)
        q = 2;

    cudaMalloc(&(frame.data[0]), frame.linesize[0] * frame.height);
    cudaMalloc(&(frame.data[1]), frame.linesize[1] * frame.height / q);
    cudaMalloc(&(frame.data[2]), frame.linesize[2] * frame.height / q);
    cudaMalloc(&(frame.data[3]), frame.linesize[3] * frame.height);
}

void AVF_GPU_wrapper::copy_to_device(const AVFrame *new_frame){
    cudaMemcpy(frame.data[0], new_frame->data[0], frame.linesize[0] * frame.height, cudaMemcpyHostToDevice);
    if (frame.linesize[1]) cudaMemcpy(frame.data[1], new_frame->data[1], frame.linesize[1] * frame.height / q, cudaMemcpyHostToDevice);
    if (frame.linesize[2]) cudaMemcpy(frame.data[2], new_frame->data[2], frame.linesize[2] * frame.height / q, cudaMemcpyHostToDevice);
    if (frame.linesize[3]) cudaMemcpy(frame.data[3], new_frame->data[3], frame.linesize[3] * frame.height, cudaMemcpyHostToDevice);
}

void AVF_GPU_wrapper::copy_to_host(const AVFrame *new_frame){
                           cudaMemcpy( new_frame->data[0],frame.data[0], frame.linesize[0] * frame.height, cudaMemcpyDeviceToHost);
    if (frame.linesize[1]) cudaMemcpy( new_frame->data[1],frame.data[1], frame.linesize[1] * frame.height / q, cudaMemcpyDeviceToHost);
    if (frame.linesize[2]) cudaMemcpy( new_frame->data[2],frame.data[2], frame.linesize[2] * frame.height / q, cudaMemcpyDeviceToHost);
    if (frame.linesize[3]) cudaMemcpy( new_frame->data[3],frame.data[3], frame.linesize[3] * frame.height, cudaMemcpyDeviceToHost);
}

void AVF_GPU_wrapper::free_from_device(){
    cudaFree(frame.data[0]);
    cudaFree(frame.data[1]);
    cudaFree(frame.data[2]);
    cudaFree(frame.data[3]);

    frame.data[0] = nullptr;
    frame.data[1] = nullptr;
    frame.data[2] = nullptr;
    frame.data[3] = nullptr;
    q = 1;
}