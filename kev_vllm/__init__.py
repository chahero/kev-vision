def register():
    from vllm import ModelRegistry
    ModelRegistry.register_model("KevQwen3_5ForPooling", "kev_vllm.model:KevQwen3_5ForPooling")
    ModelRegistry.register_model("KevQwen3_5VisionForPooling", "kev_vllm.vision:KevQwen3_5VisionForPooling")
