/* Deterministic native JPEG fixture generator, not a runtime image operation. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <jpeglib.h>
int main(int argc,char **argv) {
    if(argc!=3 || (strcmp(argv[1],"cmyk") && strcmp(argv[1],"ycck")))return 2;
    FILE *file=fopen(argv[2],"wb");if(!file)return 3;
    struct jpeg_compress_struct ctx;struct jpeg_error_mgr error;
    ctx.err=jpeg_std_error(&error);jpeg_create_compress(&ctx);jpeg_stdio_dest(&ctx,file);
    ctx.image_width=17;ctx.image_height=73;ctx.input_components=4;ctx.in_color_space=JCS_CMYK;
    jpeg_set_defaults(&ctx);jpeg_set_colorspace(&ctx,strcmp(argv[1],"cmyk")?JCS_YCCK:JCS_CMYK);jpeg_set_quality(&ctx,94,TRUE);jpeg_start_compress(&ctx,TRUE);
    unsigned char bytes[17*4];
    while(ctx.next_scanline<ctx.image_height) {
        for(unsigned x=0;x<17;x++)for(unsigned c=0;c<4;c++)bytes[x*4+c]=(unsigned char)((x*(17+c*7)+ctx.next_scanline*(31+c*11))&255);
        JSAMPROW row=bytes;jpeg_write_scanlines(&ctx,&row,1);
    }
    jpeg_finish_compress(&ctx);jpeg_destroy_compress(&ctx);return fclose(file)?4:0;
}
