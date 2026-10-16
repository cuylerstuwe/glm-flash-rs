# Required third-party notices

These notices describe code and dependency licensing, not project development history or endorsements. Retain them when redistributing the relevant portions. See [LICENSING.md](LICENSING.md) for the undecided license of original code.

| Portion | Attribution / terms |
| --- | --- |
| CUDA sampling and native RoCE implementation portions | Copyright 2026 Jay Leaton; [Apache 2.0](LICENSES/Apache-2.0.txt). See the scoped [native component notice](LICENSES/Native-Components-NOTICE.txt) and modification descriptions in source. |
| Sampling log/exp implementation and coefficient tables | Arm Optimized Routines; [MIT](LICENSES/Arm-MIT.txt). These reproduce the selected AArch64 math behavior. |
| RoCE transport and proxy adaptations in `native/roce*` | Copyright 2026 Luke Alonso and the b12x contributors; Apache 2.0. Required attribution and modification descriptions remain in source. |
| EXL3 CUDA routines in `native/exl3.cu` | ExLlamaV3, Copyright 2025 Turboderp; [MIT](LICENSES/ExLlamaV3-MIT.txt). |
| Model chat template and model-related material | Z.AI; [MIT notice](LICENSES/GLM-MIT.txt). Model weights are not included. |
| Image-processing adaptations | [Pillow license](LICENSES/Pillow.txt); ATen/PyTorch-derived resize behavior retains the [PyTorch notice](LICENSES/PyTorch.txt). |
| Vendored JPEG headers | [libjpeg-turbo terms](native/image-codec-headers/LIBJPEG-LICENSE.md) and [IJG notice](native/image-codec-headers/README.ijg). |
| Vendored WebP headers | [COPYING](native/image-codec-headers/WEBP-COPYING) and [PATENTS](native/image-codec-headers/WEBP-PATENTS). |

These terms apply to the identified third-party portions, not to independently implemented Rust host logic or to algorithms merely used as inspiration. Copyright notices for incorporated code remain scoped to that code.

Cargo dependencies are resolved by `Cargo.lock` and retain their individual licenses. XGrammar core, DFlash2 model artifacts, CUDA, NCCL, ATen/LibTorch, RDMA libraries, and precompiled GPU kernels are external dependencies; obtain their applicable notices with the exact artifacts used. This repository does not bundle their binaries or grant rights over them.
