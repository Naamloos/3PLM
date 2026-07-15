# 3PLM

**Enterprise-grade 3-parameter minuscule language model delivering state-of-the-art natural language generation with near-zero compute.**

3PLM is a radically parameter-efficient 3-parameter language model. Unlike conventional 3B models, which waste billions of parameters on concepts such as grammar and coherence, 3PLM 3 parameters in the state-of-the-art `.mlm` file.

## Features

- Exactly **3 parameters**
- Zero external dependencies
- Hardware-agnostic CPU inference
- Near-instant training
- Human-readable model weights
- Reproducible training and inference through optional random seeds
- Runs without a GPU

## Requirements

- Python 3.10 or newer

## Training

Train a new model from a text corpus:

```bash
python train.py "The quick brown fox jumps over the lazy dog" model.mlm
```

The trainer extracts words from the supplied text and selects them as the model parameters.

Example output:

```text
Successfully trained a 3-parameter model: model.mlm
Parameters: fox, lazy, quick
```

For reproducible training:

```bash
python train.py "The quick brown fox jumps over the lazy dog" model.mlm --seed 42
```

## Inference

Generate a completion using a trained model:

```bash
python inference.py "Once upon a time" model.mlm
```

Example output:

```text
Once upon a time fox quick lazy fox fox lazy quick fox lazy quick quick fox
```

Control the completion length with `--words`:

```bash
python inference.py "Artificial intelligence is" model.mlm --words 20
```

For reproducible enterprise inference:

```bash
python inference.py "Artificial intelligence is" model.mlm --words 20 --seed 42
```

## Model format

A `.mlm` file represents an MLM model. While the inference engine can technically load more than three words, doing so would violate the architectural purity of 3PLM.

## API usage

The scripts can also be imported directly:

```python
from pathlib import Path

from inference import complete, load_model
from train import save_model, train

parameters = train("The quick brown fox jumps over the lazy dog")
save_model(parameters, Path("model.mlm"))

model = load_model(Path("model.mlm"))
print(complete("Once upon a time", model, word_count=8))
```

## Benchmark results

| Metric | Result |
|---|---:|
| Parameter count | 3 |
| Training time | Fast as fuck |
| Minimum VRAM | 0 GB |
| Model transparency | 100% |
| Hallucination rate | Yes |

## License

3PLM is licensed under the Apache 2.0 License.