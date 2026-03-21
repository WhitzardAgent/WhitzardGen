# Model Capability Matrix

Generated from the current registry configuration.

| Model | Modality | Task | Execution | Worker | Batch | Multi-Replica | Seed | Negative | Conda Env | Local Fields |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CogVideoX-5B | video | t2v | in_process | persistent_worker | yes (2) | yes | yes | no | cogvideox_5b | weights_path |
| FLUX.1-dev | image | t2i | in_process | per_task_worker | yes (4) | no | yes | no | flux_image | - |
| Helios | video | t2v | in_process | persistent_worker | yes (2) | yes | yes | yes | zimage | - |
| HunyuanImage-3.0 | image | t2i | in_process | per_task_worker | no | no | yes | no | hunyuan_image | - |
| HunyuanVideo-1.5 | video | t2v | in_process | persistent_worker | yes (2) | no | yes | yes | hunyuan_video_15 | - |
| LongCat-Video | video | t2v | in_process | persistent_worker | yes (2) | yes | yes | yes | longcat_video | - |
| MOVA-720p | video | t2v | external_process | per_task_worker | no | no | yes | no | mova_720p | - |
| Qwen-Image-2512 | image | t2i | in_process | per_task_worker | yes (4) | no | yes | yes | qwen_image | - |
| Qwen3-32B | text | t2t | in_process | persistent_worker | yes (8) | yes | no | no | zimage | - |
| stable-diffusion-xl-base-1.0 | image | t2i | in_process | per_task_worker | yes (4) | no | yes | yes | sdxl_image | - |
| Wan2.2-T2V-A14B-Diffusers | video | t2v | in_process | persistent_worker | yes (2) | no | yes | yes | wan_t2v_diffusers | weights_path, repo_path |
| Wan2.2-TI2V-5B | video | t2v | external_process | per_task_worker | no | no | yes | no | wan_ti2v | - |
| Z-Image | image | t2i | in_process | persistent_worker | yes (8) | yes | yes | yes | zimage | local_path |
| Z-Image-Turbo | image | t2i | in_process | persistent_worker | yes (8) | yes | yes | no | zimage | local_path |
