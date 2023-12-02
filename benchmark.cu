#include "myconv.h"
#include "myconv_inter.h"

#include <vector>
#include <stdlib.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <chrono>

using std::chrono::milliseconds;

int main(int argc, char *argv[]){
    if (argc != 5){
        printf("bad input\n <width> <height> <in_name> <out_name>\n");
        return 1;
    }

    int width = atoi(argv[1]);
    int height = atoi(argv[2]);

    std::ifstream fin(argv[3], std::ifstream::ate | std::ifstream::binary);
    std::ofstream fout1("out_without_intermediate.r10k", std::ofstream::binary);
    std::ofstream fout2("out_with_intermediate.r10k", std::ofstream::binary);
    std::ofstream reference("reference.r10k", std::ofstream::binary);

    assert (width && height && fin && fout1 && fout2 && reference);
    size_t in_size = vc_get_datalen(width, height, RGBA);

    fin.seekg (0, std::ifstream::beg);

    std::vector<char> fin_data(in_size);
    fin.read(fin_data.data(), in_size);


    //convert RGB -> R10k -> yuv420
    std::vector<char> r10k_vec(vc_get_datalen(width, height, R10k));
    auto conv = get_decoder_from_to(RGBA, R10k);
    conv(reinterpret_cast<unsigned char *>(r10k_vec.data()),
         reinterpret_cast<const unsigned char *>(fin_data.data()), r10k_vec.size(), DEFAULT_R_SHIFT, DEFAULT_G_SHIFT, DEFAULT_B_SHIFT);


    struct to_lavc_vid_conv *to_av_conv = to_lavc_vid_conv_init(R10k, width, height, AV_PIX_FMT_YUV420P10LE, 1);
    AVFrame *converted = to_lavc_vid_conv(to_av_conv, r10k_vec.data());

    std::vector<char> converted_from_av(vc_get_datalen(width, height, R10k));

    // timing
    cudaEvent_t start1, stop1, start2, stop2;
    cudaEventCreate(&start1);
    cudaEventCreate(&stop1);

    cudaEventCreate(&start2);
    cudaEventCreate(&stop2);

    //alloc the dest buffer on gpu
    char *dst;
    cudaMalloc(&dst,(width *height *MAX_BPS + MAX_PADDING) * sizeof(char));

    int rgb_shift[] = DEFAULT_RGB_SHIFT_INIT;
    cudaFree(0);


    /* time the conversion without intermediate */

    float count_gpu1 = 0;
    for (int i = 0; i < 1000; ++i){
        cudaEventRecord(start1, 0);
        yuvp10le_to_rgb(420, dst, converted, width, height, vc_get_linesize(width, R10k), (int*) rgb_shift, 30);
        cudaMemcpy(converted_from_av.data(), dst, converted_from_av.size(), cudaMemcpyDeviceToHost);
        cudaEventRecord(stop1, 0);
        cudaEventSynchronize(stop1);
        float time1;
        cudaEventElapsedTime(&time1, start1, stop1);
        count_gpu1 += time1;
    }
    count_gpu1 /= 1000.0;

    /* write the result to file */
    fout1.write(converted_from_av.data(), converted_from_av.size());


    /* reset the converted pic */
    cudaMemset(dst, 0, width * vc_get_linesize(width, R10k));


    if (!from_lavc_init(converted, R10k))
        return -1;
    auto func = get_conversion_from_lavc(AV_PIX_FMT_YUV420P10LE, R10k);


    /* time the conversion with intermediate */
    float count_gpu2 = 0;
    for (int i = 0; i < 1000; ++i){
        cudaEventRecord(start2, 0);
        func(converted_from_av.data(), converted);
        cudaEventRecord(stop2, 0);
        cudaEventSynchronize(stop2);
        float time2;
        cudaEventElapsedTime(&time2, start2, stop2);
        count_gpu2 += time2;
    }
    count_gpu2 /= 1000.0;

    from_lavc_destroy();


    /*write the result to file*/
    fout2.write(converted_from_av.data(), converted_from_av.size());


    /* time the cpu implementation */
    std::vector<char> reference_vec(vc_get_datalen(width, height, R10k));

    float count = 0;
    for (int i = 0; i < 1000; ++i){
        auto t1 = std::chrono::high_resolution_clock::now();
        auto from_conv = get_av_to_uv_conversion(AV_PIX_FMT_YUV420P10LE, R10k);
        av_to_uv_convert(&from_conv, (char *)reference_vec.data(), converted, width, height, vc_get_linesize(width, R10k) , rgb_shift);
        auto t2 = std::chrono::high_resolution_clock::now();
        count += (t2-t1).count();
    }
    count /= 1000.0;

    reference.write(reference_vec.data(), converted_from_av.size());

    //print time
    std::cout << "time without intermediate: "  << std::fixed  << std::setprecision(10) << count_gpu1 << "ms\n"
              << "time with intermediate: " << std::fixed  << std::setprecision(10) << count_gpu2 << "ms\n"
              << "cpu implementation time: " << std::fixed  << std::setprecision(10) << count / 1000'000.0<< "ms\n";

    cudaFree(dst);
}