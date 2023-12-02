#ifndef MYCONV_INTER
#define MYCONV_INTER

#include "../../../../usr/include/x86_64-linux-gnu/libavutil/pixfmt.h"
#include "../src/config_unix.h"
#include "../src/video_codec.h"


#include "../src/libavcodec/from_lavc_vid_conv.h"
#include "../src/libavcodec/to_lavc_vid_conv.h"
#include "../src/libavcodec/lavc_common.h"
#include "../src/video_codec.h"

typedef void (*conv_t)(char * __restrict dst, const AVFrame* frame);


conv_t get_conversion_from_lavc(AVPixelFormat from, codec_t to);

bool from_lavc_init(const AVFrame*, codec_t);

void from_lavc_destroy();

#endif