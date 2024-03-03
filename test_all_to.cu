#include "to_lavc.h"
#include "from_lavc.h"

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

const std::vector<AVPixelFormat> out_codecs = {
//    AV_PIX_FMT_YUV420P10LE,
//    AV_PIX_FMT_YUV444P10LE,
//    AV_PIX_FMT_YUV422P10LE,
//    AV_PIX_FMT_P010LE,
//    AV_PIX_FMT_NV12,
//    AV_PIX_FMT_YUV420P,
//    AV_PIX_FMT_YUV422P,
//    AV_PIX_FMT_YUV444P,
//    AV_PIX_FMT_YUVJ420P,
//    AV_PIX_FMT_YUVJ422P,
//    AV_PIX_FMT_YUVJ444P,
//    AV_PIX_FMT_YUV420P12LE,
//    AV_PIX_FMT_YUV422P12LE,
//    AV_PIX_FMT_YUV444P12LE,
//    AV_PIX_FMT_YUV420P16LE,
//    AV_PIX_FMT_YUV422P16LE,
//    AV_PIX_FMT_YUV444P16LE,
//    AV_PIX_FMT_AYUV64LE,
//    AV_PIX_FMT_GBRP,
//    AV_PIX_FMT_GBRAP,
//    AV_PIX_FMT_GBRP12LE,
//    AV_PIX_FMT_GBRP10LE,
//    AV_PIX_FMT_GBRP16LE,
//    AV_PIX_FMT_GBRAP12LE,
//    AV_PIX_FMT_GBRAP10LE,
//    AV_PIX_FMT_GBRAP16LE,
//    AV_PIX_FMT_RGB24,
//    AV_PIX_FMT_RGB48LE,
//    AV_PIX_FMT_RGBA64LE,
//    AV_PIX_FMT_RGBA,
    AV_PIX_FMT_Y210,
};

const std::vector<codec_t> in_codecs = {R10k,RGB,RGBA,RG48,UYVY,YUYV,R12L,v210,Y216,Y416,};

void benchmark(int width, int height, codec_t UG_format, AVPixelFormat AV_format, unsigned char *data, std::ofstream &logs){
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

    // timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    //convert UG -> AV
    //-------------------------------------------gpu version
    AVFrame *frame1 = nullptr;
    char *dst_cpu1 = nullptr;
    float count_gpu = 0;

    if (to_lavc_init(AV_format, UG_format, width, height)){
        for (int i = 0; i < 100; ++i){
            cudaEventRecord(start, 0);
            frame1 = convert_to_lavc(UG_format, reinterpret_cast<char *>(UG_converted.data()));
            cudaEventRecord(stop, 0);
            cudaEventSynchronize(stop);
            float time;
            cudaEventElapsedTime(&time, start, stop);
            count_gpu += time;
        }
        count_gpu /= 100.0;

        if (from_lavc_init(frame1, UG_format, &dst_cpu1)){
            convert_from_lavc(frame1, dst_cpu1, UG_format);
            from_lavc_destroy(dst_cpu1);
        }
    }

    //-------------------------------------------cpu version
    float count = 0;
    char *dst_cpu2 = nullptr;
    int max = 0;

    struct to_lavc_vid_conv *conv_to_av = to_lavc_vid_conv_init(UG_format, width, height, AV_format, 1);
    if (conv_to_av && !(AV_format == AV_PIX_FMT_Y210 && (UG_format == RG48 || UG_format == Y216))){ //UG crashes here for some reason
        AVFrame *frame2 = nullptr;
        for (int i = 0; i < 100; ++i){
            auto t1 = std::chrono::high_resolution_clock::now();
            frame2 = to_lavc_vid_conv(conv_to_av, (char *) UG_converted.data());
            auto t2 = std::chrono::high_resolution_clock::now();
            count += (t2-t1).count();
        }
        count /= 100.0;

        frame2->format = AV_format;
        frame2->width = width;
        frame2->height = height;
        if (from_lavc_init(frame2, UG_format, &dst_cpu2)){
            convert_from_lavc(frame2, dst_cpu2, UG_format);

            uint8_t *f1, *f2;
            f1 = (uint8_t *)dst_cpu1;
            f2 = (uint8_t *)dst_cpu2;
            for(int i = 0; i < vc_get_datalen(width, height, UG_format); ++i) {
                max = std::max(std::abs( f1[i] - f2[i]), max);
            }
            //test validity against ug
            logs << "maximum difference against ultragrid implementation: " << max << "\n";
            from_lavc_destroy(dst_cpu2);
        }

    } else {
        logs << "non-existing cpu implementation\n";
    }

//    print time
    logs << "gpu implementation time: "  << std::fixed  << std::setprecision(10) << count_gpu << "ms\n"
         << "cpu implementation time: " << std::fixed  << std::setprecision(10) << count / 1'000'000.0<< "ms"
         << (count_gpu > count / 1'000'000.0 && count != 0 ? " <------ !!!\n" : "\n");
    logs << cudaGetErrorString(cudaGetLastError()) << "\n";


    //test validity against original
    int max2 = 0;
    uint8_t *final1 =(uint8_t *) dst_cpu1;

    for (int i = 0; i < vc_get_datalen(width, height, UG_format); ++i) {
        max2 = std::max(std::abs(final1[i] - UG_converted.data()[i]), max2);
    }
    logs << "maximum difference against original picture:" << max << (max > 1 ? " <---------- !!!\n" : "\n");
    logs.flush();

    to_lavc_destroy();
    to_lavc_vid_conv_destroy(&conv_to_av);
    free(dst_cpu1);
    free(dst_cpu2);
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
    std::ofstream fout1("conv_data", std::ofstream::binary);

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

    for (auto in_codec: in_codecs)
    {
        for (auto out_codec: out_codecs){
            std::cout << get_codec_name(in_codec) << " --> "
                      << av_get_pix_fmt_name(out_codec) << "\n";
            fout1 << get_codec_name(in_codec) << " --> "
                      << av_get_pix_fmt_name(out_codec) << "\n";
            benchmark(width, height, in_codec, out_codec,rg48vec.data(), fout1);
            std::cout << cudaGetErrorString(cudaGetLastError()) << "\n"
                      << "---------------------------------------------\n";
            fout1 << "---------------------------------------------\n";
        }
    }
}
