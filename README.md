# Query Intent Classifier

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Accuracy: 97.6%](https://img.shields.io/badge/accuracy-97.6%25-brightgreen.svg)]()
[![Model Size: 9.6KB](https://img.shields.io/badge/model%20size-9.6KB-blue.svg)]()

A tiny, multilingual classifier that detects whether a search query is a **natural language question** or a **keyword search**. Works in 50+ languages with 97.6% accuracy.

**[Live Demo](https://lukaesch.github.io/query-intent-classifier/)** | **[Blog Post](#)**

## Why?

Search systems often need to handle queries differently based on intent:

| Query Type | Example | Optimal Strategy |
|------------|---------|------------------|
| **Question** | "What are the AI trends in 2026?" | Semantic/vector search |
| **Keyword** | "Elon Musk interview" | Keyword/text search |

This classifier detects intent with **zero additional latency** by reusing embeddings you're already computing for semantic search.

## Features

- **97.6% accuracy** on held-out test set
- **9.6 KB model** - just 384 weights + bias
- **50+ languages** supported (inherits from multilingual embeddings)
- **Zero latency overhead** - reuses existing query embeddings
- **No dependencies** for inference (just numpy or raw Python)

## Quick Start

### Python

```python
import numpy as np
import json

# Load the tiny model (9.6 KB)
with open('model_weights.json') as f:
    model = json.load(f)

weights = np.array(model['weights'])
bias = model['bias']

def classify(query_embedding):
    """Classify query intent from its embedding."""
    score = np.dot(query_embedding, weights) + bias
    is_question = score > 0
    confidence = 1 / (1 + np.exp(-score))  # sigmoid
    return {
        'intent': 'question' if is_question else 'keyword',
        'confidence': float(confidence)
    }

# Example usage (assuming you have an embedding)
embedding = your_embedding_model.encode("What is machine learning?")
result = classify(embedding)
# {'intent': 'question', 'confidence': 0.97}
```

### Rust

```rust
use serde::Deserialize;

#[derive(Deserialize)]
struct Model {
    weights: Vec<f32>,
    bias: f32,
}

fn classify(embedding: &[f32], model: &Model) -> (&'static str, f32) {
    let score: f32 = embedding.iter()
        .zip(&model.weights)
        .map(|(e, w)| e * w)
        .sum::<f32>() + model.bias;

    let confidence = 1.0 / (1.0 + (-score).exp());
    let intent = if score > 0.0 { "question" } else { "keyword" };

    (intent, confidence)
}
```

### JavaScript

```javascript
async function loadModel() {
    const response = await fetch('model_weights.json');
    return await response.json();
}

function classify(embedding, model) {
    let score = model.bias;
    for (let i = 0; i < model.weights.length; i++) {
        score += embedding[i] * model.weights[i];
    }
    const confidence = 1 / (1 + Math.exp(-score));
    return {
        intent: score > 0 ? 'question' : 'keyword',
        confidence
    };
}
```

## How It Works

### The Core Insight

We use a **multilingual sentence embedding model** (paraphrase-multilingual-MiniLM-L12-v2) that maps text to 384-dimensional vectors. In this space:

- Questions cluster together (semantically similar)
- Keywords cluster separately

A simple linear classifier finds the optimal hyperplane separating them:

```
score = dot(embedding, weights) + bias
intent = "question" if score > 0 else "keyword"
```

### Why Multilingual Works

The embedding model was trained on parallel text in 50+ languages. It maps semantically equivalent queries to similar vectors regardless of language:

```
"What is AI?"     → [0.2, 0.4, ...]  ─┐
"Was ist KI?"     → [0.2, 0.4, ...]  ─┼─ Similar vectors
"¿Qué es la IA?"  → [0.2, 0.4, ...]  ─┘
```

Since all questions cluster together, our classifier works across languages automatically.

### Zero Latency

If you're already computing query embeddings for semantic search, classification adds only ~0.001ms (a single dot product):

```
Query → Embedding Model → embedding ─┬→ Semantic Search
                                     └→ Intent Classifier (384 multiplies)
```

## Training

The model was trained on ~1,200 examples:

| Source | Count | Description |
|--------|-------|-------------|
| Synthetic questions | 500 | Templates in 12 languages |
| Synthetic keywords | 500 | Common search patterns |
| Production queries | 163 | Real queries from [audioscrape.com](https://audioscrape.com) |

### Retrain on Your Data

```bash
# 1. Add your queries to real_queries.txt (one per line)

# 2. Generate training data
python generate_training_data.py

# 3. Generate embeddings (requires embedding service on localhost:5001)
python generate_embeddings.py

# 4. Train classifier
python train_classifier.py

# Output: model_weights.json (9.6 KB)
```

## Model Details

| Property | Value |
|----------|-------|
| Architecture | Logistic Regression |
| Input dimensions | 384 (from MiniLM embeddings) |
| Output | Binary (question/keyword) |
| Parameters | 385 (384 weights + 1 bias) |
| Model size | 9.6 KB (JSON) |
| Accuracy | 97.6% |
| Precision (question) | 0.98 |
| Recall (question) | 1.00 |

### Embedding Model

This classifier is designed to work with:
- **Model**: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- **Dimensions**: 384
- **Languages**: 50+

## Use Cases

### 1. Hybrid Search Optimization

```python
result = classify(query_embedding)

if result['intent'] == 'question':
    # Boost semantic search weight
    semantic_weight = 0.8
    keyword_weight = 0.2
else:
    # Favor exact keyword matching
    semantic_weight = 0.3
    keyword_weight = 0.7
```

### 2. Query Routing

```python
if classify(embedding)['intent'] == 'question':
    return answer_with_rag(query)
else:
    return search_index(query)
```

### 3. Analytics

Track what percentage of your users ask questions vs. search for keywords.

## Used By

- [audioscrape.com](https://audioscrape.com) - Podcast search engine with 10M+ transcribed segments

## Contributing

Contributions welcome! Ideas:

- Add more languages to training data
- Benchmark against other classification methods
- Create bindings for other languages (Go, Ruby, etc.)

## License

MIT License - see [LICENSE](LICENSE)

## Citation

```bibtex
@software{query_intent_classifier,
  author = {Schmyrczyk, Lukas},
  title = {Query Intent Classifier},
  year = {2025},
  url = {https://github.com/lukaesch/query-intent-classifier}
}
```
