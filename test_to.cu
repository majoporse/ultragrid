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
using std::chrono::milliseconds;
using namespace std::string_literals;



int main(int argc, char *argv[]){
    if (argc != 6){
        printf("bad input\n <width> <height> <in_name> <in_codec> <out_codec>\n");
        return 1;
    }

    int width = atoi(argv[1]);
    int height = atoi(argv[2]);
    codec_t UG_codec = get_codec_from_file_extension(argv[4]);
    AVPixelFormat AV_codec = av_get_pix_fmt(argv[5]);
    assert(AV_codec != AV_PIX_FMT_NONE && UG_codec != VIDEO_CODEC_NONE);

    std::ifstream fin(argv[3], std::ifstream::binary);
    std::ofstream fout1("AVtest_"s + argv[5] + ".rgb", std::ofstream::binary);
    std::ofstream reference("AVreference_"s + argv[5] + ".rgb", std::ofstream::binary);
    assert (width && height && fin && fout1 && reference);

    size_t in_size = vc_get_datalen(width, height, RGB);
    std::vector<unsigned char> fin_data(in_size);
    fin.read(reinterpret_cast<char *>(fin_data.data()), in_size);
//    for(auto& a: fin_data){ a = (rand() + 1) % 254; }

    //RGB -> RG48 because it has conversion to every UG format
    std::vector<unsigned char> rg48vec(vc_get_datalen(width, height, RG48));
    auto d = get_decoder_from_to(RGB, RG48);
    for (int y = 0; y < height; ++y){
        d(rg48vec.data() + y * vc_get_linesize(width, RG48),
               fin_data.data()+ y * vc_get_linesize(width, RGB),
               vc_get_linesize(width, RG48), 0, 8, 16);
    }

    //rg48 -> ug codec
    auto decode = get_decoder_from_to(RG48, UG_codec);
    if (decode == NULL){
        std::cout << "cannot find RG48 -> UG format";
        return 1;
    }
    std::vector<unsigned char> UG_converted(vc_get_datalen(width, height, UG_codec));
    for (int y = 0; y < height; ++y){
        decode(UG_converted.data() + y * vc_get_linesize(width, UG_codec),
               rg48vec.data() + y * vc_get_linesize(width, RG48),
               vc_get_linesize(width, UG_codec), 0, 8, 16);
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

    if (to_lavc_init(AV_codec, UG_codec, width, height)){
        for (int i = 0; i < 100; ++i){
            cudaEventRecord(start, 0);
            frame1 = convert_to_lavc(UG_codec, reinterpret_cast<char *>(UG_converted.data()));
            cudaEventRecord(stop, 0);
            cudaEventSynchronize(stop);
            float time;
            cudaEventElapsedTime(&time, start, stop);
            count_gpu += time;
        }
        count_gpu /= 100.0;

        if (from_lavc_init(frame1, RGB, &dst_cpu1)){
            convert_from_lavc(frame1, dst_cpu1, RGB);
        }

    } else {
        std::cout << "non-existing gpu implementation\n";
    }

    //-------------------------------------------cpu version
    float count = 0;
    char *dst_cpu2 = nullptr;
    int max = 0;
    struct to_lavc_vid_conv *conv_to_av = to_lavc_vid_conv_init(UG_codec, width, height, AV_codec, 1);

    if (conv_to_av){
        AVFrame *frame2 = nullptr;
        for (int i = 0; i < 100; ++i){
            auto t1 = std::chrono::high_resolution_clock::now();
            frame2 = to_lavc_vid_conv(conv_to_av, reinterpret_cast<char *>(UG_converted.data())); //rg48->y210 segfault!!!
            auto t2 = std::chrono::high_resolution_clock::now();
            count += (t2-t1).count();
        }
        count /= 100.0;

        frame2->format = AV_codec; //these are not set inside the UG call
        frame2->width = width;
        frame2->height = height;
        if (from_lavc_init(frame2, RGB, &dst_cpu2)){
            std::cout << "ptr " << dst_cpu2;
            convert_from_lavc(frame2, dst_cpu2, RGB);

            uint8_t *f1, *f2;
            f1 = (uint8_t *)dst_cpu1;
            f2 = (uint8_t *)dst_cpu2;
            for(int i = 0; i < vc_get_datalen(width, height, RGB); ++i) {
                max = std::max(std::abs( f1[i] - f2[i]), max);
            }
            //test validity against ug
            std::cout << "maximum difference against ultragrid implementation: " << max << "\n";

        }
    } else {
        std::cout << "non-existing cpu implementation\n";
    }

    //--------------------------------

    fout1.write(dst_cpu1, vc_get_datalen(width, height, RGB));
    reference.write(dst_cpu2, vc_get_datalen(width, height, RGB));

    //print time
    std::cout << "gpu implementation time: " << std::fixed  << std::setprecision(10) << count_gpu << "ms\n"
              << "cpu implementation time: " << std::fixed  << std::setprecision(10) << count / 1000'000.0<< "ms\n";
    std::cout << cudaGetErrorString(cudaGetLastError()) << "\n";

    int max2 = 0;
    uint8_t *final1 =(uint8_t *) dst_cpu1;

//    for (int i = 0; i < vc_get_datalen(width, height, RGB); ++i) {
//        max2 = std::max(std::abs(final1[i] - fin_data.data()[i]), max2);
//    }
    //test validity against original
    std::cout << "maximum difference against original picture:" << max2 << "\n";

    from_lavc_destroy(dst_cpu2);
    from_lavc_destroy(dst_cpu1);
    to_lavc_destroy();
}
