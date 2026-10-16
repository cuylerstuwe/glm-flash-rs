/* Native version metadata only; never imports Python or initializes CUDA. */
#include <stdint.h>
#include <stdio.h>
extern uint32_t avcodec_abi(void);
extern uint32_t avcodec_jpeg_abi(void);
extern uint32_t avcodec_webp_version(void);
int main(void) {
    printf("{\"abi\":%u,\"jpeg_abi\":%u,\"webp_version\":%u}\n",
           avcodec_abi(), avcodec_jpeg_abi(), avcodec_webp_version());
    return avcodec_abi() == 1 && avcodec_jpeg_abi() == 62 ? 0 : 1;
}
