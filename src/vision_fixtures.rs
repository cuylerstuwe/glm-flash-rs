// Exact model metadata subset for synthetic component fixtures.
pub(super) const CONFIG: &str = r#"{"image_token_id":154854,"image_start_token_id":154830,"image_end_token_id":154831,"vision_config":{
    "attention_bias": true,
    "attention_dropout": 0.0,
    "depth": 24,
    "hidden_act": "silu",
    "hidden_size": 1024,
    "image_size": 448,
    "in_channels": 3,
    "initializer_range": 0.02,
    "intermediate_size": 4096,
    "model_type": "glm5_next_vision",
    "num_heads": 16,
    "out_hidden_size": 4096,
    "patch_size": 14,
    "projection_intermediate_size": 10240,
    "rms_norm_eps": 1e-05,
    "spatial_merge_size": 2,
    "swiglu_limit": 10.0,
    "temporal_patch_size": 2
  }}"#;
pub(super) const PROCESSOR: &str = r#"{"image_processor":{
    "do_rescale": true,
    "patch_expand_factor": 1,
    "merge_size": 2,
    "image_mean": [
      0.48145466,
      0.4578275,
      0.40821073
    ],
    "image_std": [
      0.26862954,
      0.26130258,
      0.27577711
    ],
    "temporal_patch_size": 2,
    "patch_size": 14,
    "min_image_tokens": 16,
    "max_image_tokens": 8000,
    "image_processor_type": "Glm5NextImageProcessor"
  }}"#;
