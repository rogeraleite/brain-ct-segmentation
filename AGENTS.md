# Guia Para Agentes

Este projeto é um portfólio de segmentação de lesões em CT cerebral. Combina PyTorch para treino/inferência, FastAPI para expor o endpoint de segmentação e Docker para empacotar a API.

O repositório Git válido é esta pasta: `medical-imaging-portfolio/`. A pasta acima (`nicolab-study/`) é apenas um container local e não deve ser tratada como raiz do projeto.

---

## Antes De Alterar

- Leia `README.md` e `docs/PROJECT_MAP.md` para entender o fluxo principal.
- Rode `git status --short` e preserve qualquer mudança local existente.
- Não sobrescreva `models/best_model_small3DUNet.pth` sem pedido explícito — ele é o checkpoint usado pela API e pode ser caro de recriar (50 epochs, ~40–60min no Apple Silicon).
- Não altere nem versione dados em `data/raw/` ou `data/processed/`.
- Trate notebooks como artefatos de exploração; coloque lógica reutilizável em `src/`, `api/` ou `scripts/`.
- Prefira mudanças pequenas, alinhadas à estrutura atual do projeto.

---

## Comandos Canônicos

### Instalação

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

> `torch` não tem versão fixada no `requirements.txt` — compatível com Python 3.13. Se o ambiente já tem torch instalado, não force reinstalação.

### API local

```bash
# Executar sempre do diretório raiz do projeto (PYTHONPATH depende disso)
PYTHONPATH=. uvicorn api.main:app --reload --port 8000
```

> `PYTHONPATH=.` é obrigatório para os imports `src.*` funcionarem fora do Docker. Sem isso, `from src.models.unet import Small3DUNet` falhará silenciosamente com `ModuleNotFoundError`.

### Docker

```bash
# Requer models/best_model_small3DUNet.pth presente antes do build
docker compose up --build
```

> O Dockerfile copia `models/best_model_small3DUNet.pth` em tempo de build (linha 15). Se o arquivo não existir, o build falha.

### Health check

```bash
curl http://localhost:8000/health
# Esperado: {"status":"ok","model_loaded":true,"device":"mps"}
```

### Inferência

```bash
curl -X POST http://localhost:8000/segment \
  -F "file=@data/raw/images/algum_scan.nii.gz"
```

> Aceita `.nii` e `.nii.gz`. Retorna 400 para outros formatos, 413 para arquivos > 500 MB, 503 se o checkpoint não foi carregado.

### Treino

```bash
python scripts/train.py \
  --data-root data/raw \
  --save-path models/best_model_small3DUNet.pth \
  --epochs 50 \
  --batch-size 2 \
  --lr 1e-3 \
  --val-split 0.2 \
  --seed 42
# Flag opcional: --no-augment (desativa depth flip durante treino)
```

> Device selecionado automaticamente: MPS → CUDA → CPU. Tempo estimado: ~40–60min no Apple Silicon, ~2–4h no CPU.

### Pré-processamento offline (opcional)

```bash
# Converte NIfTI brutos para numpy arrays — evita re-processamento a cada época
python scripts/preprocess.py --data-root data/raw --out-dir data/processed
```

### Conversão JPG→NIfTI (específico do dataset Kaggle)

```bash
python scripts/convert_jpg_to_nifti.py \
  --dataset-dir /caminho/para/Patients_CT \
  --out-dir data/raw
```

---

## Testes Automatizados

**Não há testes automatizados configurados atualmente.** O diretório `tests/` existe mas está vazio.

Validações disponíveis sem testes formais:

```bash
# Verificação rápida de importações e shapes
python -c "
from src.data.loader import build_index, train_val_split
from src.data.dataset import BrainCTDataset
from src.models.unet import Small3DUNet
from torch.utils.data import DataLoader
import torch
m = Small3DUNet()
x = torch.zeros(1, 1, 64, 128, 128)
out = m(x)
assert out.shape == (1, 1, 64, 128, 128), f'Shape errado: {out.shape}'
assert 0 <= out.min().item() <= out.max().item() <= 1.0
print('OK — modelo funciona, shapes corretos')
"
```

```bash
# Verificação rápida da API (requer modelo carregado)
curl http://localhost:8000/health
```

---

## Cuidados Com Artefatos Grandes E Sensíveis

- `data/raw/` e `data/processed/` podem conter imagens médicas e devem continuar fora do versionamento (`.gitignore` cobre ambos).
- `models/best_model_small3DUNet.pth` é o checkpoint esperado pela API — único `.pth` comitado (os demais estão ignorados pelo `.gitignore` com exceção explícita).
- `models/best_model_small3DUNet.pth` contém `{epoch, model_state_dict, val_dice, val_loss}` — não substitua por um checkpoint de estrutura diferente sem atualizar `api/inference.py`.
- Evite rodar treino ou pré-processamento completo como validação padrão — depende de dados locais e pode demorar horas.
- `batch_size=1` quebra o `BatchNorm3d` — use mínimo 2.

---

## Cuidados Específicos Do Código

- **PYTHONPATH**: Localmente, sempre `PYTHONPATH=.`. No Docker, já está definido como `ENV PYTHONPATH=/app`.
- **Emparelhamento de dados**: `build_index()` emparelha `images/` e `masks/` por ordem alfabética. Nomes de arquivo devem ser idênticos nos dois diretórios.
- **Augmentation**: O `BrainCTDataset` flipa apenas no eixo de profundidade (D). Nunca flipa H/W — isso confundiria a detecção de lateralidade (hemisférios), que depende do eixo W.
- **MODEL_PATH em `api/inference.py`**: Hardcoded como `"models/best_model_small3DUNet.pth"` (relativo ao CWD). Rode a API sempre da raiz do projeto.
- **Se modelo não encontrado na startup**: A API sobe normalmente, mas `/segment` retorna 503. O log avisa com `Model file not found`.
- **`trainer.py` tem mudança local não comitada**: Remoção de `verbose=True` do `ReduceLROnPlateau` (deprecated no PyTorch atual). Não reverta essa mudança.

---

## Checklist Final

Antes de finalizar qualquer tarefa:

- Rode `git status --short`.
- Informe quais arquivos foram alterados.
- Informe quais validações foram executadas.
- Não há testes automatizados — declare esse risco explicitamente quando relevante.
- Mencione qualquer dependência de dados locais, checkpoint ou ambiente GPU/MPS/CUDA.
- Se o contrato da API mudar (`api/schemas.py` ou endpoints em `api/main.py`), sincronize a seção **API Reference** do `README.md`.
