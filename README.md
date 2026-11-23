# Does the Visual Backbone Matter? Pretraining Paradigms in DINO-WM

A controlled comparison of frozen visual encoders (DINOv2, CLIP, SigLIP, MAE) inside
[DINO-WM](https://github.com/gaoyuezhou/dino_wm), a world model that predicts future patch features
and plans directly in feature space. Only the encoder changes; predictor, planner, data and training
budget are fixed. Planning success is measured on two 2D tasks (PushT, Wall) and on a 3D pushing
task we added (PushCube, built on ManiSkill3).

Course project for Computer Vision and Cognitive Systems, Università degli Studi di Modena e Reggio
Emilia. Full write-up: [report](dino_wm/docs/report.pdf), [appendix](dino_wm/docs/supplementary.pdf),
[slides](dino_wm/docs/presentazione.pdf).

## Main findings

1. **Most backbones work well, but the best one depends on the task.** On PushT the backbones are
   close; the one large gap is MAE on Wall (0.44 against 0.84-0.88 for the others).
2. **On the 3D task the differences mostly disappear.** On PushCube no backbone exceeds 0.43.
3. **A lower prediction loss did not guarantee better planning.** From 5k to 10k rollouts the
   validation loss drops for every backbone, yet success falls for SigLIP on PushT (0.72 → 0.50)
   and on average on PushCube. The cause remains open (see the report).

## Repository layout

```
dino_wm/
  train.py, plan.py           training and planning entry points (from DINO-WM, modified)
  conf/                       Hydra configs; conf/encoder/{dino,dino_base,clip,siglip,mae}.yaml
  models/hf_encoders.py       CLIP, SigLIP and MAE encoder wrappers
  env/pushcube/               ManiSkill3 PushCube wrapper with the PushT/Wall interface
  datasets/pushcube_dset.py   PushCube dataset loader
  scripts/                    PushCube dataset generation
```

## Reproducing

Environment setup and the PushT and Wall datasets follow the [upstream instructions](dino_wm/README.md).
Our runs used `torch` 2.5.1, `transformers` 4.57.6, `accelerate` 1.14.0, `safetensors` 0.8.0 and
`gym` 0.26.2; PushCube also used `mani_skill` 3.0.1, `sapien` 3.0.3 and `gymnasium` 1.3.0. From
`dino_wm/`:

```bash
export DATASET_DIR=$PWD/data
```

```bash
python scripts/generate_pushcube_dataset.py --demo-path <demos.h5> --out-dir data/pushcube_noise --num-train-rollouts 10000 --num-val-rollouts 50 --max-steps 150 --action-noise-std 0.05
```

```bash
python train.py --config-name train.yaml env=pushcube encoder=clip frameskip=5 num_hist=3 env.dataset.n_rollout=5000 training.batch_size=32 training.epochs=20 hydra.run.dir=outputs/pushcube_clip_5000
```

```bash
python plan.py --config-name plan_pushcube.yaml ckpt_base_path=$PWD model_name=pushcube_clip_5000 model_epoch=20 seed=99
```

## Authors

Giulia Moretti and Rocco Lubino.

## Acknowledgements

Built on [DINO-WM](https://github.com/gaoyuezhou/dino_wm) (Zhou, Pan, LeCun, Pinto, 2024) and
[ManiSkill3](https://github.com/haosulab/ManiSkill).

## License

Released under the MIT license; see [LICENSE](LICENSE) and [dino_wm/LICENSE](dino_wm/LICENSE).
