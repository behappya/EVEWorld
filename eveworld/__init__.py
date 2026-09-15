"""EVE 包:暴露 runner 供 giga_train 的 runners=['eve.EveCausalTrainer'] 解析。"""
def __getattr__(name):
    if name == "EveCausalTrainer":
        from .method.eve_trainer import EveCausalTrainer
        return EveCausalTrainer
    if name == "EveLadLoraTrainer":
        from .method.eve_lad_lora_trainer import EveLadLoraTrainer
        return EveLadLoraTrainer
    if name == "EveFrontierTrainer":
        from .method.eve_frontier_trainer import EveFrontierTrainer
        return EveFrontierTrainer
    raise AttributeError(name)
