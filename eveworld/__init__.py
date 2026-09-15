"""EVEWorld package: lazily exposes trainer runners for giga_train (e.g. runners=["eveworld.EveCausalTrainer"])."""
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
