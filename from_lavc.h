#ifndef FROM_LAVC
#define FROM_LAVC

#include "libavutil/pixfmt.h"
#include "../src/config_unix.h"
#include "../src/video_codec.h"


#include "../src/libavcodec/from_lavc_vid_conv.h"
#include "../src/libavcodec/to_lavc_vid_conv.h"
#include "../src/libavcodec/lavc_common.h"
#include "../src/video_codec.h"

struct from_lavc_conv_state{
    char * ptr;
    codec_t to;
};

char * convert_from_lavc(from_lavc_conv_state state,  const AVFrame* frame);

from_lavc_conv_state from_lavc_init(const AVFrame*, codec_t);

void from_lavc_destroy(from_lavc_conv_state *);

#endif