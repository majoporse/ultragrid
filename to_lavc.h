#ifndef TO_LAVC
#define TO_LAVC

#include "../../../../usr/include/x86_64-linux-gnu/libavutil/pixfmt.h"
#include "../src/config_unix.h"
#include "../src/video_codec.h"


#include "../src/libavcodec/from_lavc_vid_conv.h"
#include "../src/libavcodec/to_lavc_vid_conv.h"
#include "../src/libavcodec/lavc_common.h"
#include "../src/video_codec.h"

typedef void (*conv_t)(char * __restrict dst, const AVFrame* frame);

bool convert_to_lavc(codec_t to, AVFrame *dst, const char* src);

bool to_lavc_init(AVPixelFormat, codec_t, int, int, AVFrame **);

void to_lavc_destroy(char *);


#endif