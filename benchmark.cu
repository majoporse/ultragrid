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

    char *dst_cpu1;
    cudaMallocHost(&dst_cpu1, vc_get_datalen(width, height, R10k));

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


    /* time the conversion without intermediate */

    float count_gpu1 = 0;
    for (int i = 0; i < 100; ++i){
        cudaEventRecord(start1, 0);
        yuvp10le_to_rgb(420, dst_gpu, converted, width, height, vc_get_linesize(width, R10k), (int*) rgb_shift, 30);
        cudaMemcpy(dst_cpu1, dst_gpu, vc_get_datalen(width, height, R10k), cudaMemcpyDeviceToHost);
        cudaEventRecord(stop1, 0);
        cudaEventSynchronize(stop1);
        float time1;
        cudaEventElapsedTime(&time1, start1, stop1);
        count_gpu1 += time1;
    }
    count_gpu1 /= 100.0;

    /* write the result to file */
    fout1.write(dst_cpu1, vc_get_datalen(width, height, R10k));


    /* reset the converted pic */
    cudaMemset(dst_gpu, 0, width * vc_get_linesize(width, R10k));

    char *dst_cpu2 = nullptr;
    if (!from_lavc_init(converted, R10k, &dst_cpu2))
        return -1;
    auto func = get_conversion_from_lavc(AV_PIX_FMT_YUV420P10LE, R10k);


    /* time the conversion with intermediate */
    float count_gpu2 = 0;
    for (int i = 0; i < 100; ++i){
        cudaEventRecord(start2, 0);
        func(dst_cpu2, converted);
        cudaEventRecord(stop2, 0);
        cudaEventSynchronize(stop2);
        float time2;
        cudaEventElapsedTime(&time2, start2, stop2);
        count_gpu2 += time2;
    }
    count_gpu2 /= 100.0;

    /*write the result to file*/
    fout2.write(dst_cpu2, vc_get_datalen(width, height, R10k));

    from_lavc_destroy(dst_cpu2);


    /* time the cpu implementation */
    std::vector<char> reference_vec(vc_get_datalen(width, height, R10k));

    float count = 0;
    for (int i = 0; i < 100; ++i){
        auto t1 = std::chrono::high_resolution_clock::now();
        auto from_conv = get_av_to_uv_conversion(AV_PIX_FMT_YUV420P10LE, R10k);
        av_to_uv_convert(&from_conv, (char *)reference_vec.data(), converted, width, height, vc_get_linesize(width, R10k) , rgb_shift);
        auto t2 = std::chrono::high_resolution_clock::now();
        count += (t2-t1).count();
    }
    count /= 100.0;

    reference.write(reference_vec.data(), reference_vec.size());

    //print time
    std::cout << "time without intermediate: "  << std::fixed  << std::setprecision(10) << count_gpu1 << "ms\n"
              << "time with intermediate: " << std::fixed  << std::setprecision(10) << count_gpu2 << "ms\n"
              << "cpu implementation time: " << std::fixed  << std::setprecision(10) << count / 1000'000.0<< "ms\n";
    cudaFreeHost(dst_cpu1);
    cudaFree(dst_gpu);
}