#include "from_lavc.h"
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
        std::cout << "Error allocating AVFrame\n";
        return nullptr;
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
    if (argc != 6){
        printf("bad input\n <width> <height> <in_name> <in_codec> <out_codec>\n");
        return 1;
    }

    int width = atoi(argv[1]);
    int height = atoi(argv[2]);
    AVPixelFormat in_codec = av_get_pix_fmt(argv[4]);
    codec_t out_codec = get_codec_from_file_extension(argv[5]);
    assert(in_codec != AV_PIX_FMT_NONE && out_codec != VIDEO_CODEC_NONE);

    std::ifstream fin(argv[3], std::ifstream::binary);
    std::ofstream fout1(std::string{"test_out."} + argv[5], std::ofstream::binary);
    std::ofstream reference(std::string{"reference."} + argv[5], std::ofstream::binary);
    assert (width && height && fin && fout1 && reference);

    size_t in_size = vc_get_datalen(width, height, RGBA);
    std::vector<char> fin_data(in_size);
    fin.read(fin_data.data(), in_size);


    //RGB -> avframe
    AVFrame *frame = get_avframe(width, height, AV_PIX_FMT_RGBA);

    av_image_fill_arrays(frame->data, frame->linesize, reinterpret_cast<const uint8_t *>(fin_data.data()),
                         AV_PIX_FMT_RGBA, width, height, 1);

    //avframe in converted codec
    AVFrame *converted = get_avframe(width, height, in_codec);
    convertFrame(frame, converted, in_codec);

    // timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    char *dst_cpu = nullptr;
    if (!from_lavc_init(converted, out_codec, &dst_cpu))
        return -1;

    /* time the conversion with intermediate */
    float count_gpu = 0;
    for (int i = 0; i < 100; ++i){
        cudaEventRecord(start, 0);
        convert_from_lavc(converted, dst_cpu, out_codec);
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        float time;
        cudaEventElapsedTime(&time, start, stop);
        count_gpu += time;
    }
    count_gpu /= 100.0;


    /* time the cpu implementation */
    std::vector<char> reference_vec(vc_get_datalen(width, height, out_codec));
    int rgb_shift[] = DEFAULT_RGB_SHIFT_INIT;

    float count = 0;
    for (int i = 0; i < 100; ++i){
        auto t1 = std::chrono::high_resolution_clock::now();
        auto from_conv = get_av_to_uv_conversion(in_codec, out_codec);
        if (!from_conv.valid){
            std::cout << "not a valid conversion for cpu\n";
            break;
        }
        av_to_uv_convert(&from_conv, (char *)reference_vec.data(), converted, width, height, vc_get_linesize(width, R10k) , rgb_shift);
        auto t2 = std::chrono::high_resolution_clock::now();
        count += (t2-t1).count();
    }
    count /= 100.0;


    /* write the results to files */
    fout1.write(dst_cpu, vc_get_datalen(width, height, out_codec));
    reference.write(reference_vec.data(), reference_vec.size());


    //print time
    std::cout << "gpu implementation time: "  << std::fixed  << std::setprecision(10) << count_gpu << "ms\n"
              << "cpu implementation time: " << std::fixed  << std::setprecision(10) << count / 1000'000.0<< "ms\n";
    std::cout << cudaGetErrorString(cudaGetLastError()) << "\n";

    //clean-up
    from_lavc_destroy(dst_cpu);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    av_frame_free(&converted);
    av_frame_free(&frame);
}

