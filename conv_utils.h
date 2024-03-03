#ifndef CONV_UTILS
#define CONV_UTILS
#include <libavutil/pixfmt.h>
#include "../src/config_unix.h"
#include "../src/video_codec.h"

#include "../src/libavcodec/from_lavc_vid_conv.h"
#include "../src/libavcodec/to_lavc_vid_conv.h"
#include "../src/libavcodec/lavc_common.h"
#include "../src/video_codec.h"

#ifdef WORDS_BIGENDIAN
#define BYTE_SWAP(x) (3 - x)
#else
#define BYTE_SWAP(x) x
#endif

struct AVF_GPU_wrapper{
    AVFrame frame;
    int q = 1;
    AVF_GPU_wrapper();

    void alloc(const AVFrame* new_frame);

    void copy_to_device(const AVFrame *new_frame);

    void copy_to_host(const AVFrame *new_frame);

    void free_from_device();
};

#endif
