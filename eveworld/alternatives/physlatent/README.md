# PhysLatent-GigaWorld (PhysicsLatent route)

Experimental package for adding physics-aware latent condition tokens to
GigaWorld-0-Video without changing the baseline transformer implementation.

> **Location.** This package now lives at `eveworld/alternatives/physlatent/`
> (import root `eveworld.alternatives.physlatent`) as the PhysicsLatent entry
> among the five alternative designs analyzed in the paper — see
> [`../README.md`](../README.md) for the route map and matched-control
> results. It was originally developed as a top-level package inside the
> `giga-world-0/` working tree and now lives at
> `eveworld/alternatives/physlatent/`, imported as
> `eveworld.alternatives.physlatent` (run entry points from the repository
> root with the repository root on `PYTHONPATH`).

## First Target

The v1 path trains a lightweight `PhysicsLatentEncoder` that produces `K=8`
tokens with the same dimension as GigaWorld text context (`1024`). The trainer
appends those tokens to `prompt_embeds` before calling the existing transformer:

```text
prompt_embeds + physics_tokens -> crossattn_emb
```

This keeps the VAE, scheduler, transformer blocks, and existing checkpoints
compatible. Auxiliary physics losses are wired but disabled by default until
pseudo labels are generated.

## Suggested Stages

1. Adapter warmup: freeze transformer, train only the physics latent encoder.
2. Aux warmup: load sidecar pseudo labels and train physics state heads.
3. Adapter + LoRA: train physics latent encoder and attention LoRA adapters.
4. Preference tuning: use PA-II raw score ranking to tune the adapter path.

## Smoke Check

From the repository root:

```bash
python -m eveworld.alternatives.physlatent.scripts.smoke_shapes
```

## Training Launch

Prepare imports and dependency overrides:

```bash
./eveworld/alternatives/physlatent/scripts/setup_physlatent_train_env.sh
```

Submit a one-step kjob smoke:

```bash
MAX_STEPS=1 CHECKPOINT_INTERVAL=1 RUN_NAME=physlatent_smoke_1step \
  ./eveworld/alternatives/physlatent/scripts/launch_physlatent_train_kjob.sh
```

Submit the default 200-step adapter warmup:

```bash
./eveworld/alternatives/physlatent/scripts/launch_physlatent_train_kjob.sh
```

The launcher defaults to `WITH_EMA=0` because the first stage freezes the
2B transformer and only trains the physics latent adapter.

## Pseudo Labels

Generate weak GR1 labels from prompt color cues and motion foreground:

```bash
./eveworld/alternatives/physlatent/scripts/run_generate_gr1_physlabels.sh
```

The default output is:

```text
/data/datasets/gagi/gr1_finetune_data/physlatent_pseudo_labels/gr1_physlabels_v1.json
```

Run auxiliary warmup against those labels:

```bash
BASE_CONFIG_MODULE=eveworld.alternatives.physlatent.configs.gr1_physlatent_aux \
  ./eveworld/alternatives/physlatent/scripts/launch_physlatent_train_kjob.sh
```

Then run LoRA coupling:

```bash
BASE_CONFIG_MODULE=eveworld.alternatives.physlatent.configs.gr1_physlatent_aux_lora \
  ./eveworld/alternatives/physlatent/scripts/launch_physlatent_train_kjob.sh
```
