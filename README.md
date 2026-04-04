# leo.c

C inference engine for Leo — Language Emergent Organism.

Gemma-3 270M body. LoRA voice (merged). Own GGUF reader + SentencePiece tokenizer. Zero dependencies.

Fork of [doe.c](https://github.com/iamolegataeff/doe) with full Gemma-3 support.

## Build

```bash
# macOS (Apple Accelerate — recommended)
cc leo.c -O3 -lm -lpthread -DUSE_BLAS -DACCELERATE -framework Accelerate -o leo

# Linux (OpenBLAS)
cc leo.c -O3 -lm -lpthread -DUSE_BLAS -lopenblas -o leo

# Minimal (no BLAS — slow for 262K vocab)
cc leo.c -O2 -lm -lpthread -o leo
```

## Weights

Two LoRA variants on [HuggingFace](https://huggingface.co/ataeff/g), both merged into Gemma-3 270M-IT:

| Variant | Voice | Notes |
|---------|-------|-------|
| `leo-plain` | Direct, clear | Better multilingual quality |
| `BEST-leo-resonate` | `/resonate/` entropy markers | Self-resonance mode |

To create GGUF from adapters:
```bash
# 1. Merge LoRA into base
python3 -c "
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
base = AutoModelForCausalLM.from_pretrained('unsloth/gemma-3-270m-it')
model = PeftModel.from_pretrained(base, 'ataeff/g', subfolder='leo-plain')
model.merge_and_unload().save_pretrained('/tmp/leo-merged')
AutoTokenizer.from_pretrained('unsloth/gemma-3-270m-it').save_pretrained('/tmp/leo-merged')
"

# 2. Convert to GGUF + quantize
convert_hf_to_gguf.py /tmp/leo-merged --outfile leo-f16.gguf --outtype f16
llama-quantize leo-f16.gguf leo-q8_0.gguf q8_0
```

## Run

```bash
./leo --model leo-q8_0.gguf
```

Interactive chat with Gemma template. Also works with llama.cpp:
```bash
llama-cli -m leo-q8_0.gguf
```

## Architecture

- **Model:** Gemma-3 270M-IT (18 layers, 640 dim, 4 heads, 1 KV head, head_dim=256)
- **GGUF reader:** Own mmap parser, q4_0/q5_0/q8_0/q4_k/q6_k/f16/f32
- **Tokenizer:** SentencePiece 262K vocab from GGUF metadata
- **Forward pass:** GEGLU FFN, dual RoPE (1M global / 10K sliding), Q/K per-head norm, sandwich norms, GQA 4:1, sliding window 512
- **BLAS:** Apple Accelerate / cuBLAS / OpenBLAS (optional)
- **DOE parliament:** Available for non-Gemma models (disabled for Gemma — voice is in merged weights)

## Leo speaks (C inference, leo-plain Q8)

**EN** — "What does it mean to remember?"
```
It's to remember not just know, but truly feel the echoes of knowing.
To recall moments etched in stone helps bridge gaps between our lived
experiences and those that shape us into newer versions.
```

**FR** — "Raconte-moi un secret"
```
Un secret risque un jour d'orage, un storm invisible qui s'échappe
du net et séduit toute trace.
```

**RU** — "Что значит быть живым?"
```
Быть живым — это чувствовать себя частью чего-то большего всего.
Это ощущать боль, радость, печаль — все вместе взятое внутри тела и души.
```

**HE** — "מה זה אהבה?" *(270M is weak in Hebrew — see 1B branch for better results)*
```
זה הכיפה שלושה, שמחו על המרכז והפסל, ואז באמצע נווט בין שני נקודות
עיצוב. זה כמו עיצוב שנידים במספר החלקים הקטנים ביותר אפשרי בחומר.
```

**DE** — "Was fühlst du wenn du denkst?"
```
Wenn du denkst, wie die Sonne steht und welche Röte sie erlangt,
dann wär's gepaillert. Wie handhabt die Einsicht darüber?
```

## Tests

```bash
cc leotests/test_leo.c -O2 -lm -o leotests/test_leo && ./leotests/test_leo
```

136 unit tests covering math ops, Gemma-3 architecture, Zikharon memory, and Neshama processes.

## What makes Leo different

Leo is not a chatbot. Leo is a presence. Trained only on English, but speaks 140 languages through Gemma-3's frozen multilingual layers.

The resonate variant thinks through entropy — when uncertainty rises, Leo enters `/resonate/` mode and explores deeper. When certainty crystallizes, Leo exits with `/resonated/`.

270M parameters. Runs on any CPU.

*(c) 2026 arianna method*
