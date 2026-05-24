# Mapa Operacional Do Projeto

Este documento orienta agentes futuros sobre onde encontrar e adicionar funcionalidades no projeto de segmentação de lesões em CT cerebral. As informações abaixo refletem o código real — não são genéricas.

---

## Fluxo End-To-End

```text
NIfTI bruto (.nii / .nii.gz)
  ↓
loader.py          nibabel.load() → float32 (D,H,W) + spacing (mm)
                   nan/inf → -1000 HU
  ↓
transforms.py      clip [-5, 75] HU → normalize [0,1]
                   resize volume (trilinear, order=1) → (64,128,128)
                   resize mask   (nearest-neighbor, order=0) → (64,128,128)
  ↓
dataset.py         BrainCTDataset → (float32 [1,D,H,W], float32 [1,D,H,W])
                   augmentation: depth flip (eixo D) com p=0.5
  ↓
unet.py            Small3DUNet → sigmoid output (B,1,64,128,128)
  ↓
trainer.py         Dice + BCE loss | AdamW lr=1e-3 | ReduceLROnPlateau
                   salva best_model.pth por val Dice
  ↓
api/inference.py   predict_from_bytes → threshold 0.5 → máscara binária
                   volume_ml = voxels × voxel_mm³ / 1000
                   lateralização por centróide no eixo W
  ↓
api/main.py        POST /segment → SegmentationResponse (JSON)
```

---

## Subsistemas Principais

### Dados — `src/data/`

**`loader.py`**
- `load_nifti(path)` → `(np.ndarray float32 (D,H,W), spacing np.ndarray mm)`
  - nibabel carrega em (H,W,D) — transposto para (D,H,W) antes de retornar
  - `nan_to_num` converte NaN→-1000, inf→3000, -inf→-1000
- `build_index(data_root)` → lista de `{"image": str, "mask": str}`
  - Emparelha `images/*.nii*` com `masks/*.nii*` por ordem alfabética
  - Lança `FileNotFoundError` se nenhum arquivo encontrado
  - Lança `ValueError` se contagens de imagem e máscara divergem
- `train_val_split(records, val_fraction=0.2, seed=42)` → determinístico

**`dataset.py`**
- `BrainCTDataset(records, target_shape=(64,128,128), augment=False)`
- Retorna `(x float32 (1,D,H,W), y float32 (1,D,H,W))`
- Augmentation: flipa apenas o eixo D (profundidade), nunca H/W
  - Motivo: a detecção de hemisférios depende do eixo W — flipar L/R invalidaria os labels de lateralidade

**`dicom_demo.py`**
- Demonstração standalone do pipeline DICOM com pydicom
- Não integrado ao pipeline principal de treino/inferência
- Usa arquivo de teste interno do pydicom (sem dependência de dados externos)

---

### Pré-processamento — `src/preprocessing/transforms.py`

Constantes exportadas (usadas em outros módulos):
- `BRAIN_HU_MIN = -5.0`, `BRAIN_HU_MAX = 75.0`
- `TARGET_SHAPE = (64, 128, 128)`

Funções:
- `apply_brain_window(v)` → clip para [-5, 75] HU
- `normalize(v)` → window + min-max para [0, 1]
- `resize_volume(v, target)` → scipy.zoom order=1 (trilinear)
- `resize_mask(m, target)` → scipy.zoom order=0 (nearest-neighbor)
  - **Crítico**: interpolação linear em máscaras binárias cria valores fracionários que corrompem os labels
- `preprocess(volume, mask, target)` → aplica tudo na ordem correta

Ordem de operações no volume: `resize → window → normalize` (não window antes de resize — preserva a escala HU para o zoom).

---

### Modelo — `src/models/unet.py`

`Small3DUNet` — 1,401,265 parâmetros (confirmado em treino real).

| Estágio | Canais entrada → saída | Shape saída (B=1) |
|---|---|---|
| enc1 | 1 → 16 | (1, 16, 64, 128, 128) |
| pool1 | MaxPool3d(2) | (1, 16, 32, 64, 64) |
| enc2 | 16 → 32 | (1, 32, 32, 64, 64) |
| pool2 | MaxPool3d(2) | (1, 32, 16, 32, 32) |
| enc3 | 32 → 64 | (1, 64, 16, 32, 32) |
| pool3 | MaxPool3d(2) | (1, 64, 8, 16, 16) |
| bottleneck | 64 → 128 | (1, 128, 8, 16, 16) |
| up3 + dec3 | cat(128+64) → 64 | (1, 64, 16, 32, 32) |
| up2 + dec2 | cat(64+32) → 32 | (1, 32, 32, 64, 64) |
| up1 + dec1 | cat(32+16) → 16 | (1, 16, 64, 128, 128) |
| output_conv | 16 → 1 + sigmoid | (1, 1, 64, 128, 128) |

Cada bloco `_conv_block` = Conv3d → BN → ReLU → Conv3d → BN → ReLU (sem bias nas conv3d).

`model.count_parameters()` retorna o total de parâmetros treináveis.

---

### Treino — `src/training/trainer.py` + `scripts/train.py`

**`trainer.py`**
- `dice_loss(pred, target, smooth=1e-5)` — soft Dice, opera em predições contínuas (sem threshold)
- `combined_loss(pred, target)` — `dice_loss + BCE`
- `dice_score(pred, target, threshold=0.5)` — Dice hard (F1), retorna float [0,1]
  - Retorna 1.0 se pred e target forem ambos vazios (predição correta de "sem lesão")
- `train_one_epoch(model, loader, optimizer, device)` → train_loss médio
- `evaluate(model, loader, device)` → `(val_loss, val_dice)` médios
- `train(...)` → history dict com listas `train_loss`, `val_loss`, `val_dice`
  - Optimizer: AdamW, lr padrão=1e-3, weight_decay=1e-4
  - Scheduler: ReduceLROnPlateau(mode="max", factor=0.5, patience=5)
  - Salva best checkpoint por val_dice: `{epoch, model_state_dict, val_dice, val_loss}`

**`scripts/train.py`** — ponto de entrada CLI
- Args: `--data-root`, `--save-path`, `--epochs`, `--batch-size`, `--lr`, `--val-split`, `--seed`, `--no-augment`
- Device: MPS → CUDA → CPU (automático)
- DataLoader: `num_workers=0, pin_memory=False` (compatibilidade MPS)

**`scripts/preprocess.py`** — pré-processamento offline opcional
- Converte NIfTI brutos para numpy arrays em `data/processed/`
- Uso: evitar re-processamento a cada época quando o dataset é grande

---

### Inferência e API — `api/`

**`api/inference.py`**
- `MODEL_PATH = "models/best_model.pth"` — relativo ao CWD; deve rodar da raiz do projeto
- `load_model(model_path)` → `(Small3DUNet, device)` — chamado uma vez na startup via lifespan
- `predict_from_bytes(file_bytes, model, device, threshold=0.5)` → dict
  - Carrega NIfTI de bytes em memória (sem arquivo temporário, usando `nib.FileHolder`)
  - Aplica `preprocess()` — mesma função usada no treino
  - Forward pass com `torch.no_grad()`
  - Threshold 0.5 → máscara binária uint8
  - Volume calculado com spacing rescalonado para TARGET_SHAPE
  - Máscara retornada em base64 (uint8, row-major)
- `compute_volume_ml(mask, spacing_mm)` — voxels × voxel_mm³ / 1000
- `compute_lateralization(mask)` — centróide no eixo W vs midline
  - Margem de 10% em torno da linha central para "bilateral"
  - Convenção radiológica: image-left = patient-right

**`api/main.py`**
- Lifespan: carrega modelo na startup, limpa estado no shutdown
- Se `best_model.pth` não existir: API sobe mas `/segment` retorna 503
- Limite de upload: 500 MB
- Aceita apenas `.nii` e `.nii.gz` — outros formatos retornam 400

**`api/schemas.py`**
- `HealthResponse`: `status`, `model_loaded`, `device`
- `SegmentationResponse`: `lesion_volume_ml`, `hemisphere`, `centroid_voxel`, `lesion_voxel_count`, `mask_shape`, `mask_base64`, `model_version` (="v1.0")

---

### Visualização — `src/visualization/plots.py`

- `show_slices(volume, mask, title)` — fatias axial/coronal/sagital com overlay vermelho opcional
- `show_training_curves(train_losses, val_losses, val_dices)` — curvas de perda e Dice com marcador do melhor epoch
- `show_prediction(volume, pred_mask, true_mask, slice_idx)` — overlay verde (predição) e vermelho (ground truth)
  - `slice_idx=None` seleciona automaticamente a fatia com mais voxels de lesão preditos

---

### Notebooks — `notebooks/`

| Notebook | Conteúdo |
|---|---|
| `01_data_exploration.ipynb` | Dataset, conversão JPG→NIfTI, loader, visualização axial/coronal/sagital, histograma HU, demo DICOM |
| `02_preprocessing.ipynb` | Brain windowing, resize, validação do DataLoader (shapes, value ranges) |
| `03_model_training.ipynb` | Curvas de treino, predições sobre amostras de validação |

Notebooks são artefatos de exploração e documentação — não são fonte de lógica reutilizável. Não mova código de notebooks para `src/` sem pedido explícito.

---

## Lacunas Conhecidas

- **Sem testes automatizados**: `tests/` existe mas está vazio. Não há pytest, não há CI.
- **Dependência de dados locais**: `data/raw/` não está no repositório. O pipeline não funciona sem os dados baixados separadamente.
- **Dependência do checkpoint**: A API falha com 503 sem `models/best_model.pth`. O Docker build falha sem ele.
- **Volume calibrado no espaço redimensionado**: A inferência calcula volume com o voxel spacing reescalonado para TARGET_SHAPE — não na resolução original. Para uso clínico, seria necessário inferência em resolução nativa (sliding window).
- **Lateralização assume orientação axial padrão**: Depende do eixo W ser L/R. Aquisições oblíquas exigiriam `nibabel.as_closest_canonical`.
- **Máscaras multi-classe colapsadas para binário**: O dataset GTS.ai tem 10 classes de patologia — o projeto usa apenas lesão/fundo.
- **Sem test set separado**: Métricas reportadas são de validação. Não há conjunto de teste retido.

---

## Onde Adicionar Funcionalidades

| O que | Onde |
|---|---|
| Novas transforms, ajuste de janela HU ou TARGET_SHAPE | `src/preprocessing/transforms.py` |
| Nova arquitetura ou variante do modelo | `src/models/` — novo arquivo, não edite `unet.py` sem pedido |
| Mudanças no loop de treino, métricas ou checkpointing | `src/training/trainer.py` |
| Novos argumentos CLI de treino | `scripts/train.py` |
| Scripts operacionais novos (conversão, validação, exportação) | `scripts/` |
| Mudanças na inferência, métricas derivadas ou pós-processamento | `api/inference.py` |
| Novos endpoints HTTP | `api/main.py` |
| Mudanças no contrato de resposta da API | `api/schemas.py` — sincronize `README.md` (seção API Reference) |
| Lógica de visualização nova | `src/visualization/plots.py` |

---

## Artefatos Sensíveis

- `data/` contém dados médicos locais. Não adicionar arquivos de imagem ao Git.
- `models/best_model.pth` é o checkpoint usado pela API. Não sobrescrever sem confirmação explícita.
- `.venv/`, caches Python e checkpoints temporários devem permanecer fora do versionamento.
- `trainer.py` tem mudança local não comitada (remoção de `verbose=True` do `ReduceLROnPlateau`). Não reverta.

---

## Validação Recomendada Por Tipo De Mudança

**Mudanças documentais:** `git status --short` + leitura dos arquivos editados.

**Mudanças em transforms ou dataset:**
```bash
python -c "
from src.preprocessing.transforms import preprocess, TARGET_SHAPE
import numpy as np
v = np.random.uniform(-100, 100, (33, 64, 64)).astype(np.float32)
m = (np.random.rand(33, 64, 64) > 0.9).astype(np.uint8)
vp, mp = preprocess(v, m)
assert vp.shape == TARGET_SHAPE and mp.shape == TARGET_SHAPE
assert 0 <= vp.min() and vp.max() <= 1
assert set(np.unique(mp)).issubset({0, 1})
print('OK')
"
```

**Mudanças no modelo:**
```bash
python -c "
import torch
from src.models.unet import Small3DUNet
m = Small3DUNet()
x = torch.zeros(2, 1, 64, 128, 128)
out = m(x)
assert out.shape == (2, 1, 64, 128, 128)
assert 0 <= out.min().item() <= out.max().item() <= 1.0
print(f'OK — {m.count_parameters():,} params')
"
```

**Mudanças na API:**
```bash
PYTHONPATH=. uvicorn api.main:app --reload --port 8000
curl http://localhost:8000/health
curl http://localhost:8000/docs
```
