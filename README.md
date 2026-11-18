# SAE-Speech-Neuroscience

Analysis code for the manuscript

**Sparse language-model features trade context for word identity in human cortical responses to speech**<br>
Amirmasoud Ahmadi and Fatemeh Ghorbani Lohesara

The code uses intracranial recordings from the Podcast ECoG dataset and the Brain Treebank, together
with Gemma-2 language-model representations and their GemmaScope sparse-autoencoder features.

## Repository layout

| Folder | Contents |
|---|---|
| `config/` | analysis parameters (`config.yaml`) and variants |
| `utils/` | configuration loading, input locations, run logging |
| `preprocessing/` | electrode table and neural targets |
| `feature_extraction/` | language-model, dictionary and nuisance features |
| `encoding/` | cross-validation and banded-ridge encoding models |
| `capacity_matching/` | comparisons of representations at equal dimensionality |
| `interpretability/` | feature coherence, ablation and profiling |
| `random_model_control/` | autoencoder training and evaluation |
| `word_types/` | word-type analyses |
| `controls/` | robustness analyses |
| `inference/` | statistics shared by the analyses |
| `film_cohort/` | Brain Treebank analyses |
| `figures/` | source data and a plotting script for each figure panel ([details](figures/README.md)) |

Each module's docstring describes its inputs and outputs.

## Installation

Python 3.11. Feature extraction and autoencoder training need a CUDA GPU.

```bash
git clone https://github.com/amirmasoud92/SAE-Speech-Neuroscience.git
cd SAE-Speech-Neuroscience
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

Gemma-2 and Llama-3.1 are gated on the Hugging Face Hub: accept their licences and run
`huggingface-cli login`.

## Data

No data are included. Place the inputs under `data/` or set the environment variables:

- Podcast ECoG dataset, [OpenNeuro ds005574](https://openneuro.org/datasets/ds005574): `data/ds005574` (`PODCAST_BIDS_ROOT`)
- Brain Treebank, [BrainTreebank.dev](https://BrainTreebank.dev): `data/braintreebank` (`BTB_ROOT`), with a word-to-neural-sample alignment table (`BTB_ALIGNMENT`)
- Concreteness and valence–arousal–dominance norms: `data/norms/`
- Language models and dictionaries are downloaded from the Hugging Face Hub on first use.

## Usage

Run modules from the repository root; they read `config/config.yaml` and write to `results/`.

```bash
python -m preprocessing.electrode_table --bids-root data/ds005574 --out results/electrodes
python -m feature_extraction.language_model_features --transcript data/ds005574/stimuli/podcast_transcript.csv --out results/features
python -m encoding.fit_encoding_model --stage all
```

Figure panels can be redrawn from the included source data, for example
`python figures/figure_4/panel_f/plot.py`.

## Citation

The citation will be added on publication. In the meantime, please cite this repository
([`CITATION.cff`](CITATION.cff)).

## Contact

Amirmasoud Ahmadi, Max Planck Institute for Biological Intelligence, Seewiesen, Germany<br>
Amirmasoud.Ahmadi@bi.mpg.de

## License

No license has been specified yet; until one is added, no permission to reuse or redistribute the
code is granted.
