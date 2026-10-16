/* Native codec primitives only. Rust owns image policy, allocations, row
 * sequencing, color conversion and lifetime. No Python or GPU dependency. */
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <setjmp.h>
#include <jpeglib.h>
#include <jerror.h>
#include <webp/decode.h>
#include <webp/demux.h>

typedef struct {
    struct jpeg_decompress_struct decoder;
    struct jpeg_error_mgr error;
    jmp_buf jump;
    char message[JMSG_LENGTH_MAX];
    int created;
    int started;
    int failed;
} AJpeg;

static void copy_error(char *out, size_t size, const char *message) {
    if (out && size) snprintf(out, size, "%s", message);
}
static void jpeg_fail(j_common_ptr ptr) {
    AJpeg *ctx = (AJpeg *)ptr;
    ctx->failed = 1;
    (*ptr->err->format_message)(ptr, ctx->message);
    longjmp(ctx->jump, 1);
}
static void jpeg_message(j_common_ptr ptr, int level) {
    /* jpeg_mem_src inserts EOI for a truncated buffer. The frozen loader does
     * not enable LOAD_TRUNCATED_IMAGES, so do not silently accept that repair. */
    if (level < 0 && ptr->err->msg_code == JWRN_JPEG_EOF) jpeg_fail(ptr);
}

uint32_t avcodec_abi(void) { return 1; }
uint32_t avcodec_jpeg_abi(void) { return JPEG_LIB_VERSION; }
uint32_t avcodec_webp_version(void) { return (uint32_t)WebPGetDecoderVersion(); }
void avcodec_dependency_symbols(const void **jpeg,const void **webp,const void **demux) {
    *jpeg=(const void *)jpeg_std_error;
    *webp=(const void *)WebPGetDecoderVersion;
    *demux=(const void *)WebPAnimDecoderNewInternal;
}

int avjpeg_open(const uint8_t *bytes, size_t size, void **output,
                uint32_t *width, uint32_t *height, uint32_t *channels,
                char *error, size_t error_size) {
    if (!bytes || !size || !output || !width || !height || !channels) return 0;
    *output = NULL;
    AJpeg *ctx = calloc(1, sizeof(*ctx));
    if (!ctx) { copy_error(error, error_size, "JPEG allocation"); return 0; }
    ctx->decoder.err = jpeg_std_error(&ctx->error);
    ctx->error.error_exit = jpeg_fail;
    ctx->error.emit_message = jpeg_message;
    if (setjmp(ctx->jump)) {
        copy_error(error, error_size, ctx->message);
        if (ctx->created) jpeg_destroy_decompress(&ctx->decoder);
        free(ctx); return 0;
    }
    ctx->created = 1;
    jpeg_create_decompress(&ctx->decoder);
    ctx->decoder.mem->max_memory_to_use = 64L * 1024L * 1024L;
    jpeg_mem_src(&ctx->decoder, bytes, size);
    if (jpeg_read_header(&ctx->decoder, TRUE) != JPEG_HEADER_OK) {
        copy_error(error, error_size, "JPEG incomplete header");
        jpeg_destroy_decompress(&ctx->decoder); free(ctx); return 0;
    }
    if (ctx->decoder.data_precision != 8 ||
        (ctx->decoder.num_components != 1 && ctx->decoder.num_components != 3 && ctx->decoder.num_components != 4)) {
        copy_error(error, error_size, "JPEG unsupported sample/component count");
        jpeg_destroy_decompress(&ctx->decoder); free(ctx); return 0;
    }
    /* The frozen non-draft path uses default ISLOW/fancy upsampling. */
    ctx->decoder.dct_method = JDCT_ISLOW;
    ctx->decoder.do_fancy_upsampling = TRUE;
    ctx->decoder.out_color_space = ctx->decoder.num_components == 4 ? JCS_CMYK : JCS_RGB;
    *width = ctx->decoder.image_width;
    *height = ctx->decoder.image_height;
    *channels = ctx->decoder.num_components == 4 ? 4 : 3;
    *output = ctx;
    return 1;
}

int avjpeg_start(void *raw, char *error, size_t error_size) {
    AJpeg *ctx = raw;
    if (!ctx || ctx->failed || ctx->started) return 0;
    if (setjmp(ctx->jump)) { copy_error(error,error_size,ctx->message); return 0; }
    if (!jpeg_start_decompress(&ctx->decoder)) { copy_error(error,error_size,"JPEG incomplete start");ctx->failed=1;return 0; }
    ctx->started=1;
    return 1;
}

int avjpeg_rows(void *raw, uint8_t *output, size_t size, uint32_t max_rows,
                uint32_t *read_rows, char *error, size_t error_size) {
    AJpeg *ctx=raw;
    if (!ctx || !output || !read_rows || !ctx->started || ctx->failed || !max_rows) return 0;
    if (setjmp(ctx->jump)) { copy_error(error,error_size,ctx->message); return 0; }
    const size_t stride=(size_t)ctx->decoder.output_width*ctx->decoder.output_components;
    uint32_t count=ctx->decoder.output_height-ctx->decoder.output_scanline;
    if (count>max_rows) count=max_rows;
    if (stride && count>size/stride) { copy_error(error,error_size,"JPEG output bound");ctx->failed=1;return 0; }
    *read_rows=0;
    while (*read_rows<count) {
        JSAMPROW row=output+(size_t)(*read_rows)*stride;
        if (jpeg_read_scanlines(&ctx->decoder,&row,1)!=1) { copy_error(error,error_size,"JPEG incomplete scanline");ctx->failed=1;return 0; }
        ++*read_rows;
    }
    return 1;
}
int avjpeg_finish(void *raw, char *error, size_t error_size) {
    AJpeg *ctx=raw;
    if (!ctx || !ctx->started || ctx->failed) return 0;
    if (setjmp(ctx->jump)) { copy_error(error,error_size,ctx->message); return 0; }
    if (!jpeg_finish_decompress(&ctx->decoder)) { copy_error(error,error_size,"JPEG incomplete finish");ctx->failed=1;return 0; }
    ctx->started=0;
    return 1;
}
void avjpeg_free(void *raw) {
    AJpeg *ctx=raw;
    if (ctx) {
        /* Even a library cleanup failure must never longjmp through Rust. */
        if(setjmp(ctx->jump)==0 && ctx->created)jpeg_destroy_decompress(&ctx->decoder);
        free(ctx);
    }
}

int avwebp_header(const uint8_t *bytes,size_t size,uint32_t *width,uint32_t *height,
                  uint32_t *alpha,char *error,size_t error_size) {
    WebPBitstreamFeatures info;
    if (!bytes || !width || !height || !alpha) return 0;
    if (WebPGetFeatures(bytes,size,&info)!=VP8_STATUS_OK) { copy_error(error,error_size,"WebP invalid header");return 0; }
    *width=(uint32_t)info.width;*height=(uint32_t)info.height;*alpha=(uint32_t)info.has_alpha;
    return 1;
}
int avwebp_open(const uint8_t *bytes,size_t size,void **output,char *error,size_t error_size) {
    if(!bytes || !size || !output)return 0;
    WebPData data={bytes,size};
    /* Matches frozen Pillow: default straight RGBA, first reconstructed canvas. */
    *output=WebPAnimDecoderNew(&data,NULL);
    if(!*output){copy_error(error,error_size,"WebP decoder creation");return 0;}
    return 1;
}
int avwebp_first(void *raw,const uint8_t **output,uint32_t *width,uint32_t *height,
                 char *error,size_t error_size) {
    if(!raw || !output || !width || !height)return 0;
    WebPAnimInfo info;
    if(!WebPAnimDecoderGetInfo(raw,&info)){copy_error(error,error_size,"WebP canvas metadata");return 0;}
    uint8_t *pixels=NULL;int timestamp=0;
    if(!WebPAnimDecoderGetNext(raw,&pixels,&timestamp)){copy_error(error,error_size,"WebP initial frame");return 0;}
    *output=pixels;*width=info.canvas_width;*height=info.canvas_height;
    return 1;
}
void avwebp_free(void *raw) { if(raw)WebPAnimDecoderDelete(raw); }
