#include "myconv.h"
#include "myconv_inter.h"
extern "C" {
#include <libswscale/swscale.h>
};

#include <vector>
#include <stdlib.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <chrono>

using std::chrono::milliseconds;

void convertFrame(AVFrame* srcFrame, AVFrame* dstFrame, enum AVPixelFormat dstFormat) {
    struct SwsContext *sws_ctx = nullptr;

    // Initialize the SwsContext for the conversion
    sws_ctx = sws_getContext(
            srcFrame->width, srcFrame->height, static_cast<AVPixelFormat>(srcFrame->format),
            dstFrame->width, dstFrame->height, dstFormat,
            SWS_BICUBIC, nullptr, nullptr, nullptr
    );

    if (!sws_ctx) {
        fprintf(stderr, "Error creating SwsContext\n");
        return;
    }

    // Perform the conversion
    sws_scale(sws_ctx, srcFrame->data, srcFrame->linesize, 0,
              srcFrame->height, dstFrame->data, dstFrame->linesize);

    // Free the SwsContext
    sws_freeContext(sws_ctx);
}

AVFrame * get_avframe(int width, int height, AVPixelFormat p){
    AVFrame *frame = av_frame_alloc();
    if (!frame) {
        fprintf(stderr, "Error allocating AVFrame\n");
        return NULL;
    }

    // Set the frame properties
    frame->width = width;
    frame->height = height;
    frame->format = p;

    // Allocate buffer for the frame
    av_frame_get_buffer(frame, 0);
    return frame;
}

int main(int argc, char *argv[]){
    if (argc != 4){
        printf("bad input\n <width> <height> <in_name>\n");
        return 1;
    }

    int width = atoi(argv[1]);
    int height = atoi(argv[2]);
    AVPixelFormat in_codec = AV_PIX_FMT_YUV420P10LE;
    codec_t out_codec = R10k;

    std::ifstream fin(argv[3], std::ifstream::ate | std::ifstream::binary);
    std::ofstream fout1("out_without_intermediate.r10k", std::ofstream::binary);
    std::ofstream fout2("out_with_intermediate.r10k", std::ofstream::binary);
    std::ofstream reference("reference.r10k", std::ofstream::binary);

    assert (width && height && fin && fout1 && fout2 && reference
            && in_codec != AV_PIX_FMT_NONE && out_codec != VIDEO_CODEC_NONE);
    size_t in_size = vc_get_datalen(width, height, RGBA);

    fin.seekg (0, std::ifstream::beg);

    std::vector<char> fin_data(in_size);
    fin.read(fin_data.data(), in_size);


    //RGB -> avframe
    AVFrame *frame = get_avframe(width, height, AV_PIX_FMT_RGBA);

    int linesize[1] = { 3 * width };
    av_image_fill_arrays(frame->data, linesize, reinterpret_cast<const uint8_t *>(fin_data.data()), AV_PIX_FMT_RGBA, width, height, 1);

    //avframe in converted codec
    AVFrame *converted = get_avframe(width, height, in_codec);
    convertFrame(frame, converted, in_codec);

    // timing
    cudaEvent_t start1, stop1, start2, stop2;
    cudaEventCreate(&start1);
    cudaEventCreate(&stop1);

    cudaEventCreate(&start2);
    cudaEventCreate(&stop2);


    //alloc the dest buffer on gpu
    char *dst_gpu;
    cudaMalloc(&dst_gpu, (width * height * MAX_BPS + MAX_PADDING) * sizeof(char));

    int rgb_shift[] = DEFAULT_RGB_SHIFT_INIT;
    cudaFree(0);

    //using page-locked memory for faster transfers
    char *dst_cpu1;
    cudaMallocHost(&dst_cpu1, vc_get_datalen(width, height, out_codec));

    /* time the conversion without intermediate */
    float count_gpu1 = 0;
    for (int i = 0; i < 100; ++i){
        cudaEventRecord(start1, 0);
        yuvp10le_to_rgb(420, dst_gpu, converted, width, height, vc_get_linesize(width, out_codec), (int*) rgb_shift, 30);
        cudaMemcpy(dst_cpu1, dst_gpu, vc_get_datalen(width, height, out_codec), cudaMemcpyDeviceToHost);
        cudaEventRecord(stop1, 0);
        cudaEventSynchronize(stop1);
        float time1;
        cudaEventElapsedTime(&time1, start1, stop1);
        count_gpu1 += time1;
    }
    count_gpu1 /= 100.0;

    /* reset the converted pic */
    cudaMemset(dst_gpu, 0, width * vc_get_linesize(width, out_codec));

    char *dst_cpu2 = nullptr;
    if (!from_lavc_init(converted, out_codec, &dst_cpu2))
        return -1;

    /* time the conversion with intermediate */
    float count_gpu2 = 0;
    for (int i = 0; i < 100; ++i){
        cudaEventRecord(start2, 0);
        convert_from_lavc(converted, dst_cpu2, out_codec);
        cudaEventRecord(stop2, 0);
        cudaEventSynchronize(stop2);
        float time2;
        cudaEventElapsedTime(&time2, start2, stop2);
        count_gpu2 += time2;
    }
    count_gpu2 /= 100.0;


    /* time the cpu implementation */
    std::vector<char> reference_vec(vc_get_datalen(width, height, out_codec));

    float count = 0;
    for (int i = 0; i < 100; ++i){
        auto t1 = std::chrono::high_resolution_clock::now();
        auto from_conv = get_av_to_uv_conversion(in_codec, out_codec);
        av_to_uv_convert(&from_conv, (char *)reference_vec.data(), converted, width, height, vc_get_linesize(width, R10k) , rgb_shift);
        auto t2 = std::chrono::high_resolution_clock::now();
        count += (t2-t1).count();
    }
    count /= 100.0;


    /* write the results to files */
    fout1.write(dst_cpu1, vc_get_datalen(width, height, out_codec));
    fout2.write(dst_cpu2, vc_get_datalen(width, height, out_codec));
    reference.write(reference_vec.data(), reference_vec.size());


    //print time
    std::cout << "time without intermediate: "  << std::fixed  << std::setprecision(10) << count_gpu1 << "ms\n"
              << "time with intermediate: " << std::fixed  << std::setprecision(10) << count_gpu2 << "ms\n"
              << "cpu implementation time: " << std::fixed  << std::setprecision(10) << count / 1000'000.0<< "ms\n";
    std::cout << cudaGetErrorString(cudaGetLastError());

    //clean-up
//    from_lavc_destroy(dst_cpu2);
    cudaFreeHost(dst_cpu1);
    cudaFree(dst_gpu);

    cudaEventDestroy(start1);
    cudaEventDestroy(start2);
    cudaEventDestroy(stop1);
    cudaEventDestroy(stop2);

    av_frame_free(&converted);
    av_frame_free(&frame);
}
