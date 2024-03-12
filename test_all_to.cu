#include "to_lavc.h"
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
#include "from_lavc.h"
#include <string>
#include <algorithm>
#include <ranges>
#include <filesystem>
#include <numeric>


using std::chrono::milliseconds;
using namespace std::string_literals;

const std::vector<std::tuple<codec_t, AVPixelFormat, int>> convs = {
            { v210, AV_PIX_FMT_YUV420P10LE, 2},
            { v210, AV_PIX_FMT_YUV422P10LE, 2},
            { v210, AV_PIX_FMT_YUV444P10LE, 2},
            { v210, AV_PIX_FMT_YUV444P16LE, 2},
#if Y210_PRESENT
            { v210, AV_PIX_FMT_Y210LE, 2},
            { Y216, AV_PIX_FMT_Y210, 2},
#endif
            { R10k, AV_PIX_FMT_YUV444P10LE, 2},
            { R10k, AV_PIX_FMT_YUV444P12LE, 2},
            { R10k, AV_PIX_FMT_YUV444P16LE, 2},
            { R12L, AV_PIX_FMT_YUV444P10LE, 2},
            { R12L, AV_PIX_FMT_YUV444P12LE, 2},
            { R12L, AV_PIX_FMT_YUV444P16LE, 2},
            { RG48, AV_PIX_FMT_YUV444P10LE, 2},
            { RG48, AV_PIX_FMT_YUV444P12LE, 2},
            { RG48, AV_PIX_FMT_YUV444P16LE, 2},
            { v210, AV_PIX_FMT_P010LE, 2},

            { UYVY, AV_PIX_FMT_YUV422P, 1},
            { UYVY, AV_PIX_FMT_YUVJ422P, 1},

            { UYVY, AV_PIX_FMT_YUV420P, 1},
            { UYVY, AV_PIX_FMT_YUVJ420P, 1},
            { UYVY, AV_PIX_FMT_NV12, 1},
            { UYVY, AV_PIX_FMT_YUV444P, 1},
            { UYVY, AV_PIX_FMT_YUVJ444P, 1},
            { Y216, AV_PIX_FMT_YUV422P10LE, 2},
            { Y216, AV_PIX_FMT_YUV422P16LE, 2},
            { Y216, AV_PIX_FMT_YUV444P16LE, 2},
            { RGB, AV_PIX_FMT_BGR0, 1},
            { RGB, AV_PIX_FMT_GBRP, 1},
            { RGB, AV_PIX_FMT_YUV444P, 1},
            { RGBA, AV_PIX_FMT_GBRP, 1},
            { RGBA, AV_PIX_FMT_BGRA, 1},
            { R10k, AV_PIX_FMT_BGR0, 1},
            { R10k, AV_PIX_FMT_GBRP10LE, 2},
            { R10k, AV_PIX_FMT_GBRP16LE, 2},

            { R10k, AV_PIX_FMT_YUV422P10LE, 2},
            { R10k, AV_PIX_FMT_YUV420P10LE, 2},
            { R12L, AV_PIX_FMT_GBRP12LE, 2},
            { R12L, AV_PIX_FMT_GBRP16LE, 2},
            { RG48, AV_PIX_FMT_GBRP12LE, 2},
};


void check(AVFrame*f1, AVFrame *f2, int bpp, std::ofstream &logs){
    int q = 1;
    if (f1->format == AV_PIX_FMT_YUV420P16LE ||
            f1->format == AV_PIX_FMT_YUV420P12LE ||
            f1->format == AV_PIX_FMT_YUVJ420P ||
            f1->format == AV_PIX_FMT_YUV420P ||
            f1->format == AV_PIX_FMT_YUV420P10LE ||
            f1->format == AV_PIX_FMT_P010LE ||
            f1->format == AV_PIX_FMT_NV12)
        q = 2;
    int max = 0;
    if (bpp == 2){
        uint16_t  *d11 = (uint16_t *) f1->data[0];
        uint16_t  *d12 = (uint16_t *) f1->data[1];
        uint16_t  *d13 = (uint16_t *) f1->data[2];

        uint16_t  *d21 = (uint16_t *) f2->data[0];
        uint16_t  *d22 = (uint16_t *) f2->data[1];
        uint16_t  *d23 = (uint16_t *) f2->data[2];

        for(int i = 0; i < f1->linesize[0] / 2 * f1->height; i++)
            max = std::max(std::abs( (int) d11[i] - (int) d21[i]), max);
        for(int i = 0; i < f1->linesize[1] / 2 * f1->height / q; i++)
            max = std::max(std::abs( (int) d12[i] - (int) d22[i]), max);
        for(int i = 0; i < f1->linesize[2] / 2 * f1->height / q; i++)
            max = std::max(std::abs( (int) d13[i] - (int) d23[i]), max);

    } else {
        uint8_t  *d11 = (uint8_t *) f1->data[0];
        uint8_t  *d12 = (uint8_t *) f1->data[1];
        uint8_t  *d13 = (uint8_t *) f1->data[2];

        uint8_t  *d21 = (uint8_t *) f2->data[0];
        uint8_t  *d22 = (uint8_t *) f2->data[1];
        uint8_t  *d23 = (uint8_t *) f2->data[2];

        for(int i = 0; i < f2->linesize[0] * f2->height; i++)
            max = std::max(std::abs( (int) d11[i] - (int) d21[i]), max);
        for(int i = 0; i < f2->linesize[1] * f2->height / q; i++)
            max = std::max(std::abs( (int) d12[i] - (int) d22[i]), max);
        for(int i = 0; i < f2->linesize[2] * f2->height / q; i++)
            max = std::max(std::abs( (int) d13[i] - (int) d23[i]), max);
    }
    logs << "maximum difference against ultragrid implementation: " << max << "\n";
}

void benchmark(int width, int height, codec_t UG_format, AVPixelFormat AV_format, unsigned char *data, std::ofstream &logs, int bpp){
    //rg48 -> ug codec
    auto decode = get_decoder_from_to(RG48, UG_format);
    if (decode == NULL){
        std::cout << "cannot find RG48 -> UG format\n";
        return;
    }
    std::vector<unsigned char> UG_converted(vc_get_datalen(width, height, UG_format));
    for (int y = 0; y < height; ++y){
        decode(UG_converted.data() + y * vc_get_linesize(width, UG_format),
               data + y * vc_get_linesize(width, RG48),
               vc_get_linesize(width, UG_format), 0, 16, 32);
    }

    //-------------------------------------------cpu version
    float count = 0;
    int max = 0;

    AVFrame *frame2 = nullptr;
    struct to_lavc_vid_conv *conv_to_av = to_lavc_vid_conv_init(UG_format, width, height, AV_format, 1);
    if (conv_to_av && !(AV_format == AV_PIX_FMT_Y210 && (UG_format == RG48 || UG_format == Y216))){ //UG crashes here for some reason
        for (int i = 0; i < 10; ++i){
            auto t1 = std::chrono::high_resolution_clock::now();
            frame2 = to_lavc_vid_conv(conv_to_av, (char *) UG_converted.data());
            auto t2 = std::chrono::high_resolution_clock::now();
            count += (t2-t1).count();
        }
        count /= 10.0;

    } else {
        std::cout << "non-existing cpu implementation\n";
        to_lavc_vid_conv_destroy(&conv_to_av);
        return;
    }
    // timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    //convert UG -> AV
    //-------------------------------------------gpu version
    AVFrame *frame1 = nullptr;
    float count_gpu = 0;

    auto state = to_lavc_init(AV_format, UG_format, width, height);
    if (state.frame){
        for (int i = 0; i < 10; ++i){
            cudaEventRecord(start, 0);
            frame1 = convert_to_lavc(state, reinterpret_cast<char *>(UG_converted.data()));
            cudaEventRecord(stop, 0);
            cudaEventSynchronize(stop);
            float time;
            cudaEventElapsedTime(&time, start, stop);
            count_gpu += time;
        }
        count_gpu /= 10.0;
    } else {
        std::cout << "error";
    }

    frame2->format = frame1->format;
    frame2->width = frame1->width;
    frame2->height = frame1->height;
    frame2->linesize[0] = frame1->linesize[0];
    frame2->linesize[1] = frame1->linesize[1];
    frame2->linesize[2] = frame1->linesize[2];
    frame2->linesize[3] = frame1->linesize[3];

    logs << get_codec_name(UG_format) << " --> "
          << av_get_pix_fmt_name(AV_format) << "\n";
    //test validity against ug
    check(frame1, frame2, bpp, logs);

//    print time
    logs << "gpu implementation time: " << std::fixed  << std::setprecision(10) << count_gpu << "ms\n"
         << "cpu implementation time: " << std::fixed  << std::setprecision(10) << count / 1'000'000.0<< "ms"
         << (count_gpu > count / 1'000'000.0 && count != 0 ? " <------ !!!\n" : "\n");
    logs << cudaGetErrorString(cudaGetLastError()) << "\n";
    logs << "---------------------------------------------\n";
    logs.flush();

    to_lavc_destroy(&state);
    to_lavc_vid_conv_destroy(&conv_to_av);

    cudaEventDestroy(stop);
    cudaEventDestroy(start);
}

int main(int argc, char *argv[]){
    if (argc != 4){
        printf("bad input\n <width> <height>\n");
        return 1;
    }

    int width = atoi(argv[1]);
    int height = atoi(argv[2]);
    std::ifstream fin(argv[3], std::ifstream::binary);
    std::ofstream fout1("logs_to", std::ofstream::binary);

    size_t in_size = vc_get_datalen(width, height, RGB);
    std::vector<unsigned char> fin_data(in_size);

    std::filesystem::path p{argv[3]};
    if (std::filesystem::file_size(p) != in_size){
        std::cout << "wrong file size! make sure its 16 bit rgb!!\n";
        return 1;
    }

    fin.read(reinterpret_cast<char *>(fin_data.data()), in_size);
//    std::ranges::for_each(fin_data, [](auto &a){a = rand();});

//    RGB -> RG48 because it has conversion to every UG format
    std::vector<unsigned char> rg48vec(vc_get_datalen(width, height, RG48));
    auto d = get_decoder_from_to(RGB, RG48);
    for (int y = 0; y < height; ++y){
        d(rg48vec.data() + y * vc_get_linesize(width, RG48),
          fin_data.data()+ y * vc_get_linesize(width, RGB),
          vc_get_linesize(width, RG48), 0, 8, 16);
    }

    for (auto [in_codec, out_codec, bpp]: convs){
        std::cout << get_codec_name(in_codec) << " --> "
                  << av_get_pix_fmt_name(out_codec) << "\n";
        benchmark(width, height, in_codec, out_codec,rg48vec.data(), fout1, bpp);
        std::cout << cudaGetErrorString(cudaGetLastError()) << "\n"
                  << "---------------------------------------------\n";
    }

}
