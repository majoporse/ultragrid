#ifndef MYCONV
#define MYCONV
#include <libavutil/pixfmt.h>
#include "../src/config_unix.h"
#include "../src/video_codec.h"
//#include "conv_utils.h"

#include "../src/libavcodec/from_lavc_vid_conv.h"
#include "../src/libavcodec/to_lavc_vid_conv.h"
#include "../src/libavcodec/lavc_common.h"
#include "../src/video_codec.h"

void yuvp10le_to_rgb(int subsampling, char * __restrict dst_buffer, AVFrame *frame,
                     int width, int height, int pitch, const int * __restrict rgb_shift, int out_bit_depth);

#endif