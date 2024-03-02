FLAGS ?= -g
SRCDIR ?= ..
COMMON_FLAGS = $(FLAGS) -D_GNU_SOURCE -I$(SRCDIR)/src/
MKDIR_P = mkdir -p
vpath %.c $(SRCDIR) $(SRCDIR)/tools
vpath %.cpp $(SRCDIR) $(SRCDIR)/tools
vpath %.cu $(SRCDIR) $(SRCDIR)/conversions
CUDA_CXX = /usr/local/cuda-12.0/bin/nvcc
%.o : %.c
	$(MKDIR_P) $(dir $@)
	$(CC) $(COMMON_FLAGS) -c $< -o $@

%.o : %.cpp
	$(MKDIR_P) $(dir $@)
	$(CXX) -std=c++20 $(COMMON_FLAGS) -c $< -o $@

%.o : %.cu
	$(MKDIR_P) $(dir $@)
	$(CUDA_CXX) -std c++20 $(COMMON_FLAGS) -c $< -o $@  -lavutil -lavcodec -lswscale

benchmark_conv: benchmark_conv.o conv_utils.o src/debug.o src/libavcodec/from_lavc_vid_conv.o \
                                src/libavcodec/lavc_common.o src/libavcodec/to_lavc_vid_conv.o \
                                src/libavcodec/utils.o src/pixfmt_conv.o src/utils/color_out.o \
                                src/utils/misc.o src/utils/pam.o src/utils/parallel_conv.o \
                                src/utils/thread.o src/utils/worker.o src/utils/y4m.o src/video_codec.o \
                                src/video_frame.o
	$(CXX) $^ -o benchmark_conv -lavutil -lavcodec

benchmark: benchmark.o myconv.o myconv_inter.o src/debug.o src/libavcodec/from_lavc_vid_conv.o \
                                src/libavcodec/lavc_common.o src/libavcodec/to_lavc_vid_conv.o \
                                src/libavcodec/utils.o src/pixfmt_conv.o src/utils/color_out.o \
                                src/utils/misc.o src/utils/pam.o src/utils/parallel_conv.o \
                                src/utils/thread.o src/utils/worker.o src/utils/y4m.o src/video_codec.o \
                                src/video_frame.o
	$(CUDA_CXX) $^  -o benchmark -lavutil -lavcodec -lswscale

test_from: test_from.o from_lavc.o src/debug.o src/libavcodec/from_lavc_vid_conv.o \
                                src/libavcodec/lavc_common.o src/libavcodec/to_lavc_vid_conv.o \
                                src/libavcodec/utils.o src/pixfmt_conv.o src/utils/color_out.o \
                                src/utils/misc.o src/utils/pam.o src/utils/parallel_conv.o \
                                src/utils/thread.o src/utils/worker.o src/utils/y4m.o src/video_codec.o \
                                src/video_frame.o
	$(CUDA_CXX) $^  -o test_from  -lavutil -lavcodec -lswscale

test_to: test_to.o from_lavc.o to_lavc.o src/debug.o src/libavcodec/from_lavc_vid_conv.o \
                                src/libavcodec/lavc_common.o src/libavcodec/to_lavc_vid_conv.o \
                                src/libavcodec/utils.o src/pixfmt_conv.o src/utils/color_out.o \
                                src/utils/misc.o src/utils/pam.o src/utils/parallel_conv.o \
                                src/utils/thread.o src/utils/worker.o src/utils/y4m.o src/video_codec.o \
                                src/video_frame.o
	$(CUDA_CXX) $^  -o test_to  -lavutil -lavcodec -lswscale

test_all_to: test_all_to.o from_lavc.o to_lavc.o src/debug.o src/libavcodec/from_lavc_vid_conv.o \
                                src/libavcodec/lavc_common.o src/libavcodec/to_lavc_vid_conv.o \
                                src/libavcodec/utils.o src/pixfmt_conv.o src/utils/color_out.o \
                                src/utils/misc.o src/utils/pam.o src/utils/parallel_conv.o \
                                src/utils/thread.o src/utils/worker.o src/utils/y4m.o src/video_codec.o \
                                src/video_frame.o
	$(CUDA_CXX) $^  -o test_all_to  -lavutil -lavcodec -lswscale
