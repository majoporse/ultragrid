#include "from_lavc.h"
#include <map>
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

const std::vector<std::tuple<AVPixelFormat, codec_t>> in_codecs = {
        {AV_PIX_FMT_YUV420P10LE, v210},
        {AV_PIX_FMT_YUV420P10LE, UYVY},
        {AV_PIX_FMT_YUV420P10LE, RGB},
        {AV_PIX_FMT_YUV420P10LE, RGBA},
        {AV_PIX_FMT_YUV420P10LE, R10k},
        {AV_PIX_FMT_YUV422P10LE, v210},
        {AV_PIX_FMT_YUV422P10LE, UYVY},
        {AV_PIX_FMT_YUV422P10LE, RGB},
        {AV_PIX_FMT_YUV422P10LE, RGBA},
        {AV_PIX_FMT_YUV422P10LE, R10k},
        {AV_PIX_FMT_YUV444P10LE, v210},
        {AV_PIX_FMT_YUV444P10LE, UYVY},
        {AV_PIX_FMT_YUV444P10LE, R10k},
        {AV_PIX_FMT_YUV444P10LE, RGB},
        {AV_PIX_FMT_YUV444P10LE, RGBA},
        {AV_PIX_FMT_YUV444P10LE, R12L},
        {AV_PIX_FMT_YUV444P10LE, RG48},
        {AV_PIX_FMT_YUV444P10LE, Y416},
        {AV_PIX_FMT_P010LE, v210},
        {AV_PIX_FMT_P010LE, UYVY},
        // 8-bit YUV
        {AV_PIX_FMT_YUV420P, v210}, //ug bad
        {AV_PIX_FMT_YUV420P, UYVY},
        {AV_PIX_FMT_YUV420P, RGB},
        {AV_PIX_FMT_YUV420P, RGBA},
        {AV_PIX_FMT_YUV422P, v210},
        {AV_PIX_FMT_YUV422P, UYVY},
        {AV_PIX_FMT_YUV422P, RGB},
        {AV_PIX_FMT_YUV422P, RGBA},
        {AV_PIX_FMT_YUV444P, v210},
        {AV_PIX_FMT_YUV444P, UYVY},
        {AV_PIX_FMT_YUV444P, RGB},
        {AV_PIX_FMT_YUV444P, RGBA},

        {AV_PIX_FMT_YUVJ420P, v210}, //ug bad
        {AV_PIX_FMT_YUVJ420P, UYVY},
        {AV_PIX_FMT_YUVJ420P, RGB},
        {AV_PIX_FMT_YUVJ420P, RGBA},
        {AV_PIX_FMT_YUVJ422P, v210},
        {AV_PIX_FMT_YUVJ422P, UYVY},
        {AV_PIX_FMT_YUVJ422P, RGB},
        {AV_PIX_FMT_YUVJ422P, RGBA},
        {AV_PIX_FMT_YUVJ444P, v210},
        {AV_PIX_FMT_YUVJ444P, UYVY},
        {AV_PIX_FMT_YUVJ444P, RGB},
        {AV_PIX_FMT_YUVJ444P, RGBA},
        // 8-bit YUV 2)
        {AV_PIX_FMT_NV12, UYVY},
        {AV_PIX_FMT_NV12, RGB},
        {AV_PIX_FMT_NV12, RGBA},
        // 12-UV
        {AV_PIX_FMT_YUV444P12LE, R10k},
        {AV_PIX_FMT_YUV444P12LE, R12L},
        {AV_PIX_FMT_YUV444P12LE, RG48},
        {AV_PIX_FMT_YUV444P12LE, UYVY},
        {AV_PIX_FMT_YUV444P12LE, v210}, //ug bad
        {AV_PIX_FMT_YUV444P12LE, Y416},
        // 16-bit YUV
        {AV_PIX_FMT_YUV444P16LE, R10k},
        {AV_PIX_FMT_YUV444P16LE, R12L},
        {AV_PIX_FMT_YUV444P16LE, RG48},
        {AV_PIX_FMT_YUV444P16LE, UYVY},
        {AV_PIX_FMT_YUV444P16LE, v210}, //ug bad
        {AV_PIX_FMT_YUV444P16LE, Y416},
        {AV_PIX_FMT_AYUV64LE, UYVY}, //somehow crashed
        {AV_PIX_FMT_AYUV64, v210}, //ug bad
        {AV_PIX_FMT_AYUV64, Y416},
        // RGB
        {AV_PIX_FMT_GBRAP, RGB},
        {AV_PIX_FMT_GBRAP, RGBA},
        {AV_PIX_FMT_GBRP, RGB},
        {AV_PIX_FMT_GBRP, RGBA},
        {AV_PIX_FMT_RGB24, UYVY},
        {AV_PIX_FMT_RGB24, RGBA},
        {AV_PIX_FMT_GBRP10LE, R10k},
        {AV_PIX_FMT_GBRP10LE, RGB},
        {AV_PIX_FMT_GBRP10LE, RGBA},
        {AV_PIX_FMT_GBRP10LE, RG48},
        {AV_PIX_FMT_GBRP12LE, R12L},
        {AV_PIX_FMT_GBRP12LE, R10k},
        {AV_PIX_FMT_GBRP12LE, RGB},
        {AV_PIX_FMT_GBRP12LE, RGBA},
        {AV_PIX_FMT_GBRP12LE, RG48},
        {AV_PIX_FMT_GBRP16LE, R12L},
        {AV_PIX_FMT_GBRP16LE, R10k},
        {AV_PIX_FMT_GBRP16LE, RG48},
        {AV_PIX_FMT_GBRP12LE, RGB},
        {AV_PIX_FMT_GBRP12LE, RGBA},
        {AV_PIX_FMT_RGB48LE, R12L},
        {AV_PIX_FMT_RGB48LE, RGBA},
};
const std::map<codec_t, codec_t> convs = {{R10k, RG48}, {R12L, RG48}, {v210, RG48}};

bool convertFrame(AVFrame* srcFrame, AVFrame* dstFrame, enum AVPixelFormat dstFormat) {
    struct SwsContext *sws_ctx = nullptr;

    // Initialize the SwsContext for the conversion
    sws_ctx = sws_getContext(
            srcFrame->width, srcFrame->height, static_cast<AVPixelFormat>(srcFrame->format),
            dstFrame->width, dstFrame->height, dstFormat,
            SWS_BICUBIC, nullptr, nullptr, nullptr
    );

    if (!sws_ctx) {
        fprintf(stderr, "Error creating SwsContext\n");
        return false;
    }

    // Perform the conversion
    sws_scale(sws_ctx, srcFrame->data, srcFrame->linesize, 0,
              srcFrame->height, dstFrame->data, dstFrame->linesize);

    // Free the SwsContext
    sws_freeContext(sws_ctx);
    return true;
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

void check(uint8_t *p1, uint8_t *p2, int width, int height, codec_t format, std::ofstream &logs){
    uint8_t * p3 = nullptr;
    uint8_t * p4 = nullptr;
    //convert
    if (convs.contains(format)){
        auto *decode = get_decoder_from_to(format, RG48);
        p3 = (uint8_t *) malloc(vc_get_datalen(width, height, RG48));
        p4 = (uint8_t *) malloc(vc_get_datalen(width, height, RG48));

        for (int y = 0; y < height; ++y) {
            decode(reinterpret_cast<unsigned char *>(&p3[y * vc_get_linesize(width, RG48)]),
                   reinterpret_cast<unsigned char *>(&p1[y * vc_get_linesize(width, format)]),
                   vc_get_linesize(width, RG48), 0, 8, 16);
            decode(reinterpret_cast<unsigned char *>(&p4[y * vc_get_linesize(width, RG48)]),
                   reinterpret_cast<unsigned char *>(&p2[y * vc_get_linesize(width, format)]),
                   vc_get_linesize(width, RG48), 0, 8, 16);
        }
        p1 = p3;
        p2 = p4;
        format = convs.at(format);
    }

    int max = 0;
    if (format == RG48 || format == Y216 || format == Y416){
        auto d1 = (uint16_t *) p1;
        auto d2 = (uint16_t *) p2;
        for (int i = 0; i < vc_get_datalen(width, height, format) / 2; ++i) {
            max = std::max(std::abs((int) d1[i] - (int) d2[i]), max);
            if (max > 60) {std::cout << " " << i << " "; break;}
        }
    } else{
        for (int i = 0; i < vc_get_datalen(width, height, format); ++i){
            max = std::max(std::abs( (int) p1[i] - (int) p2[i]), max);
        }
    }

    logs << "maximum difference against ultragrid implementation: " << max << "\n";
    if (p1 == p3) free(p3);
    if (p1 == p3) free(p4);
}

void benchmark(AVFrame *f1, AVPixelFormat AV_format, codec_t UG_format, std::ofstream &logs){
    //avframe in converted codec
    int width = f1->width;
    int height = f1->height;
    AVFrame *converted = get_avframe(f1->width, f1->height, AV_format);
    if (!convertFrame(f1, converted, AV_format)){
        av_frame_free(&converted);
        return;
    }

    //---------------------------------------cpu implementation
    std::vector<char> reference_vec(vc_get_datalen(width, height, UG_format));
    int rgb_shift[] = DEFAULT_RGB_SHIFT_INIT;

    float count = 0;
    auto from_conv = get_av_to_uv_conversion(AV_format, UG_format);
    if (!from_conv){
        std::cout << "not a valid conversion for cpu\n";
        av_frame_free(&converted);
        return;
    }

    for (int i = 0; i < 10; ++i){
        auto t1 = std::chrono::high_resolution_clock::now();
        av_to_uv_convert(from_conv, (char *)reference_vec.data(), converted, width, height, vc_get_linesize(width, UG_format) , rgb_shift);
        auto t2 = std::chrono::high_resolution_clock::now();
        count += (t2-t1).count();
    }
    count /= 10.0;


    //---------------------------------------gpu implementation
    char *dst_cpu = nullptr;
    if (!from_lavc_init(converted, UG_format, &dst_cpu)){
        return;
    }

    // timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float count_gpu = 0;
    for (int i = 0; i < 10; ++i){
        cudaEventRecord(start, 0);
        convert_from_lavc(converted, dst_cpu, UG_format);
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        float time;
        cudaEventElapsedTime(&time, start, stop);
        count_gpu += time;
    }
    count_gpu /= 10.0;

    logs << av_get_pix_fmt_name(AV_format) << " --> "
         << get_codec_name(UG_format) << "\n";
    check((uint8_t *) dst_cpu,(uint8_t *) reference_vec.data(), width, height, UG_format, logs);

    logs << "gpu implementation time: " << std::fixed  << std::setprecision(10) << count_gpu << "ms\n"
         << "cpu implementation time: " << std::fixed  << std::setprecision(10) << count / 1000'000.0<< "ms\n"
         << cudaGetErrorString(cudaGetLastError()) << "\n"
         << "----------------------------------------------\n";

    //clean-up
    std::cout << "A "; std::cout.flush();
    av_frame_free(&converted);
    from_lavc_destroy(&dst_cpu);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

}

int main(int argc, char *argv[]){
    if (argc != 4){
        printf("bad input\n <width> <height> <in_name> <in_codec> <out_codec>\n");
        return 1;
    }

    int width = atoi(argv[1]);
    int height = atoi(argv[2]);

    std::ifstream fin(argv[3], std::ifstream::binary);
    std::ofstream fout1("logs_from", std::ofstream::binary);
    assert (width && height && fin && fout1);

    size_t in_size = vc_get_datalen(width, height, RG48);
    std::vector<char> fin_data(in_size);
//    fin.read(fin_data.data(), in_size);
    for (auto &a: fin_data){ a = rand(); }

    //RGB -> avframe
    AVFrame *frame = get_avframe(width, height, AV_PIX_FMT_RGB48LE);

    av_image_fill_arrays(frame->data, frame->linesize, reinterpret_cast<const uint8_t *>(fin_data.data()),
                         AV_PIX_FMT_RGB48LE, width, height, 1);

    for (auto [in_codec, out_codec]: in_codecs)
    {
        std::cout << av_get_pix_fmt_name(in_codec) << " --> "
                  << get_codec_name(out_codec) << "\n";
        benchmark(frame,in_codec, out_codec, fout1);
        std::cout << cudaGetErrorString(cudaGetLastError()) << "\n"
                  << "---------------------------------------------\n";
    }
}

