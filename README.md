# DiffusionTaxonomy: Probing WordNet Knowledge in Generative Models

This project investigates whether text-to-image generative models encode hierarchical knowledge consistent with linguistic ontologies. Specifically, we evaluate if Stable Diffusion's generative capabilities respect the concept hierarchies defined in WordNet's taxonomy.

The core hypothesis is that if a generative model has properly encoded conceptual knowledge, the distribution of images generated from a hypernym concept (e.g., "dog") should encompass the distributions of its hyponyms (e.g., "corgi", "poodle", "dalmatian"). This provides a systematic way to:

1. Test the comprehensiveness of a model's knowledge
2. Identify potential biases in image generation
3. Evaluate how well semantic relationships in language transfer to visual representations

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/wordnet-diffusion.git
   cd wordnet-diffusion
   ```

2. Create and activate the conda environment:
   ```bash
   conda env create -f environment.yml
   conda activate wnet-diff
   ```

3. Download the WordNet corpus for NLTK:
   ```bash
   python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"
   ```

## Usage

### WordNet Hierarchy Extraction and Visualization

To extract and visualize concept hierarchies from WordNet, use the command-line script:

```bash
python src/generate_hierarchy.py dog --output-dir hierarchies/
```

For visualizations:

```bash
python src/generate_hierarchy.py dog --output-dir hierarchies/ --visualize --layout twopi
```

### Image Generation with Stable Diffusion

Run these scripts in sequence:

```bash
python src/generate_sd_prompts.py hierarchies/dog.json --output-dir prompts/
python src/generate_sd_images.py prompts/dog_prompts.json --output-dir images/
```

## Methodology

Our approach consists of several key components:

### WordNet Hierarchy Extraction
We extract hierarchies from WordNet using the `generate_hierarchy.py` script, which allows configuring depth and frequency thresholds.

### Image Generation with Stable Diffusion
We generate images using prompts derived from the WordNet hierarchies. The process involves:
- Creating prompts with the `generate_sd_prompts.py` script
- Generating images with the `generate_sd_images.py` script

### Feature Extraction and Analysis
We extract features from generated images and perform two experiments:
1. Classification to hypernyms
2. Clustering of hyponyms

Run the analysis scripts:
```bash
python scripts/extract_features.py ./data/images --output ./data/features
python scripts/train_classifier.py ./data/features --output ./data/classifiers
python scripts/evaluate_clustering.py ./data/features --output ./data/clustering
```

### Visualization and Interpretation
Generate visualizations of the results:
```bash
python scripts/visualize_results.py ./data/features --output ./data/visualizations
```

Our experiments primarily focused on the "dog" hierarchy as a case study, but the methodology can be applied to any concept in WordNet.
