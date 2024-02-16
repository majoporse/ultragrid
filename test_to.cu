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
    std::ofstream fout1("AVtest_"s + argv[5] + ".rgba", std::ofstream::binary);
    std::ofstream reference("AVreference_"s + argv[5] + ".rgba", std::ofstream::binary);
    assert (width && height && fin && fout1 && reference);

    size_t in_size = vc_get_datalen(width, height, RGBA);
    std::vector<unsigned char> fin_data(in_size);
    fin.read(reinterpret_cast<char *>(fin_data.data()), in_size);


    //RGBA -> RG48 because it has conversion to every UG format
    std::vector<unsigned char> rg48vec(vc_get_datalen(width, height, RG48));
    auto d = get_decoder_from_to(RGBA, RG48);
    for (int y = 0; y < height; ++y){
        d(rg48vec.data() + y * vc_get_linesize(width, RG48),
               fin_data.data()+ y * vc_get_linesize(width, RGBA),
               vc_get_linesize(width, RG48), 0, 8, 16);
    }

    //rg48 -> ug codec
    auto decode = get_decoder_from_to(RG48, UG_codec);
    if (decode == NULL){
        std::cout << "cannot find RGBA -> UG format";
        return 1;
    }
    std::vector<unsigned char> UG_converted(vc_get_datalen(width, height, UG_codec));
    for (int y = 0; y < height; ++y){
        decode(UG_converted.data() + y * vc_get_linesize(width, UG_codec),
               rg48vec.data() + y * vc_get_linesize(width, RG48),
               vc_get_linesize(width, UG_codec), 0, 8, 16);
    }

    std::cout << AV_codec << '\n';
    std::cout.flush();

    //convert UG -> AV
    //-------------------------------------------gpu version
    AVFrame *frame1 = nullptr;
    char *dst_cpu1 = nullptr;
    if (to_lavc_init(AV_codec, UG_codec, width, height, &frame1)){
        convert_to_lavc(UG_codec, frame1, reinterpret_cast<char *>(UG_converted.data()));
        if (from_lavc_init(frame1, RGBA, &dst_cpu1))
            convert_from_lavc(frame1, dst_cpu1, RGBA);
    } else {
        std::cout << "non-existing gpu implementation\n";
    }

    //-------------------------------------------cpu version
    struct to_lavc_vid_conv *conv_to_av = to_lavc_vid_conv_init(UG_codec, width, height, AV_codec, 1);
    char *dst_cpu2 = nullptr;
    if (conv_to_av){
        AVFrame *frame2 = to_lavc_vid_conv(conv_to_av, (char *) UG_converted.data());
        if (from_lavc_init(frame2, RGBA, &dst_cpu2))
            convert_from_lavc(frame2, dst_cpu2, RGBA);
    } else {
        std::cout << "non-existing cpu implementation\n";
    }

    //--------------------------------

    fout1.write(dst_cpu1, vc_get_datalen(width, height, RGBA));
    reference.write(dst_cpu2, vc_get_datalen(width, height, RGBA));
    std::cout << cudaGetErrorString(cudaGetLastError());
}
