#ifndef TO_LAVC
#define TO_LAVC

#include "../../../../usr/include/x86_64-linux-gnu/libavutil/pixfmt.h"
#include "../src/config_unix.h"
#include "../src/video_codec.h"


#include "../src/libavcodec/from_lavc_vid_conv.h"
#include "../src/libavcodec/to_lavc_vid_conv.h"
#include "../src/libavcodec/lavc_common.h"
#include "../src/video_codec.h"

struct to_lavc_conv_state{
    AVFrame *frame;
    codec_t to;
};

typedef void (*conv_t)(char * __restrict dst, const AVFrame* frame);

AVFrame *convert_to_lavc(to_lavc_conv_state, const char* src);

to_lavc_conv_state to_lavc_init(AVPixelFormat, codec_t, int, int);

void to_lavc_destroy(to_lavc_conv_state*);


#endif