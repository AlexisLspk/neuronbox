# Multi-GPU avec NeuronBox (documentation)

NeuronBox **n’orchestre pas** encore l’entraînement multi-GPU (pas de lanceur DDP intégré). Ce document décrit le **contrat actuel** et comment enchaîner avec PyTorch / l’écosystème.

## Exposer plusieurs GPU au processus

- **`neuron run --gpu 0,1`** (ou `--gpu 0,1,2`) : la CLI définit `CUDA_VISIBLE_DEVICES` pour le script d’entrée.
- Vous pouvez aussi fixer **`CUDA_VISIBLE_DEVICES`** vous-même dans l’environnement ou dans `neuron.yaml` → section `env`.

Les identifiants sont ceux vus par le pilote NVIDIA sur la machine hôte (hors conteneur, ou dans le conteneur si vous passez par Docker avec `--gpus all` / device requests).

## Entraînement distribué (DDP, DeepSpeed, etc.)

La responsabilité du **lancement** (nombre de processus, `MASTER_ADDR`, ports, etc.) reste dans votre **`entrypoint`** (souvent un shell ou un script Python qui appelle `torchrun`).

Exemple minimal (à adapter à votre projet) :

```bash
# Après avoir résolu le venv / deps avec neuron.yaml
torchrun --nproc_per_node=2 train.py
```

Ou avec variables explicites :

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 train.py
```

NeuronBox fournit aujourd’hui surtout : **environnement Python hashé**, **store modèle**, **enregistrement session** auprès du daemon (VRAM soft, stats). Il ne remplace pas `torchrun`, Slurm ou Kubernetes.

## Champ `gpu.strategy` dans `neuron.yaml`

Le schéma accepte `single` | `pipeline` | `tensor`. **Aucune orchestration automatique** n’est appliquée aujourd’hui : le champ sert de **documentation / intention** et pourra piloter plus tard des lanceurs ou des validations.

Feuille de route possible :

1. Valider la cohérence `strategy` vs nombre de GPU visibles.
2. Générer ou suggérer une ligne de commande `torchrun` / documentée.
3. Intégration optionnelle avec un scheduler (hors scope « local seul »).

## Isolation forte (MIG, partitions)

Pour un **quota VRAM matériel** par job, voir [GPU_VRAM.md](GPU_VRAM.md) (MIG, `CUDA_VISIBLE_DEVICES` sur UUID d’instance).

## macOS et Linux

- **Linux** : workflow CUDA / ROCm classique ; multi-GPU NVIDIA via `CUDA_VISIBLE_DEVICES` + votre lanceur.
- **macOS** : souvent un seul device Apple / Metal ; le multi-GPU « datacenter » ne s’applique généralement pas.

## Voir aussi

- [GPU_VRAM.md](GPU_VRAM.md) — quotas soft, MIG, monitoring.
- [neuronbox-mvp.md](../neuronbox-mvp.md) — vision produit et historique de spécification.
