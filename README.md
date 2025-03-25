# DiffusionTaxonomy: Probing WordNet Knowledge in Generative Models

This project investigates whether text-to-image generative models encode hierarchical knowledge consistent with linguistic ontologies. Specifically, we evaluate if Stable Diffusion's generative capabilities respect the concept hierarchies defined in WordNet's taxonomy.

The core hypothesis is that if a generative model has properly encoded conceptual knowledge, the distribution of images generated from a hypernym concept (e.g., "dog") should encompass the distributions of its hyponyms (e.g., "corgi", "poodle", "dalmatian"). This provides a systematic way to:

1. Test the comprehensiveness of a model's knowledge
2. Identify potential biases in image generation
3. Evaluate how well semantic relationships in language transfer to visual representations

By systematically exploring different areas of WordNet's taxonomy and analyzing the resulting generated images, we can gain insights into how effectively text-conditional models like Stable Diffusion encode hierarchical knowledge structures.

## Installation

### Using pip

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/wordnet-diffusion.git
   cd wordnet-diffusion
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download the WordNet corpus for NLTK:
   ```bash
   python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"
   ```

### Using Conda

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
   python -c "import nltk; nltk.download('wordnet')"
   ```

## Usage

### WordNet Hierarchy Extraction and Visualization

To extract and visualize concept hierarchies from WordNet:

```python
from src.wordnet_hierarchy_extractor import extract_hierarchy
from src.visualization_tools import visualize_hierarchy
import matplotlib.pyplot as plt
import json

# Extract a hierarchy for a concept
concept = "dog"
hier_df, graph = extract_hierarchy(
    concept,
    max_depth=2,
    frequency_threshold=1
)

# Save the hierarchy data as JSON
output_file = f"data/synset_hierarchies/{concept}.json"
# Find depth columns
depth_cols = [col for col in hier_df.columns if col.startswith('cat_depth_')]
depth_cols.sort()

# Create simplified hierarchy data
simplified_data = []
for _, row in hier_df.iterrows():
    path_names = [row[col] for col in depth_cols if pd.notna(row[col])]
    simplified_data.append({
        'synset_id': row['synset_id'],
        'name': row['class'],
        'definition': row['definition'],
        'path_names': path_names
    })

with open(output_file, 'w') as f:
    json.dump(simplified_data, f, indent=2)

# Visualize the hierarchy using different layouts
visualize_hierarchy(graph, layout="dot", figsize=(10, 15))
plt.savefig(f"outputs/visualizations/{concept}_dot.png")
plt.close()

visualize_hierarchy(graph, layout="twopi", figsize=(15, 15))
plt.savefig(f"outputs/visualizations/{concept}_twopi.png")
plt.close()
```

#### Command-line Interface

You can also use the provided command-line script to generate hierarchies:

Basic usage:

```bash
python scripts/generate_hierarchy.py dog --output data/synset_hierarchies/dog.json
```

This will create a WordNet hierarchy for the concept "dog" with a depth of 2 and save it as a JSON file.

Advanced usage:

```bash
python scripts/generate_hierarchy.py vehicle --max-depth 3 --freq-threshold 5 --output data/custom/vehicle.json --visualize --vis-dir data/visualizations --layouts dot twopi
```

For all available options:

```bash
python scripts/generate_hierarchy.py --help
```

## Methodology

Our approach consists of several key components:

### WordNet Hierarchy Extraction

We begin by selecting a root concept in WordNet (e.g., "dog") and recursively traverse its hyponyms to create a hierarchy of related concepts. To manage computational complexity:

- We limit the exploration to a maximum depth (configurable)
- For dense hierarchies, we filter synsets based on lemma frequency thresholds
- We store descriptions for each synset to enrich our prompts

Users can extract hierarchies using our command-line script or the API functions directly. This allows for quick exploration of different semantic domains within WordNet.

**Files used**: `wordnet_hierarchy_extractor.py`, `synset_processor.py`

### Image Generation with Stable Diffusion

For each synset in our hierarchy:

1. We construct prompts by concatenating the synset name with its description
2. We append "photograph" to each prompt to improve image quality
3. We employ Composable-Diffusion to combine the full prompt with a version using only the synset name
4. We generate multiple images per synset using consistent parameters

To generate images using our scripts:

```bash
# First, extract a WordNet hierarchy and save it as a JSON file
python scripts/generate_hierarchy.py dog --output ./data/dog.json

# Then, generate prompts from the hierarchy data
python scripts/generate_sd_prompts.py ./data/dog.json --output ./data

# You can also generate prompts only for leaf nodes (specific categories)
python scripts/generate_sd_prompts.py ./data/dog.json --output ./data --leaves-only

# Finally, generate images using the Stable Diffusion API
python scripts/generate_sd_images.py ./data/prompts/dog_prompts.json --output ./data/images
```

For custom prompt formats (e.g., for dog breeds):

```bash
python scripts/generate_sd_prompts.py ./data/dog.json --format dog --output ./data
```

After grid-searching various parameters, we settled on:
- Sampler: LMS
- Sampling Steps: 35
- CFG Scale: 5.5

**Files used**: `generate_hierarchy.py`, `generate_sd_prompts.py`, `generate_sd_images.py`

### Feature Extraction and Analysis

For each generated image:

1. We extract feature vectors using a pre-trained VGG16 model
2. We reduce dimensionality using PCA for visualization
3. We conduct two main experiments:

#### Experiment 1: Classification to Hypernyms
We train classifiers on images from synsets with hyponyms and evaluate on images from their hyponyms. A high classification accuracy indicates that the distributions are properly nested according to the WordNet hierarchy.

**Files used**: `feature_extractor.py`, `classifier_trainer.py`, `evaluation_metrics.py`

#### Experiment 2: Clustering Hyponyms
We perform unsupervised clustering on images generated from hyponyms and compare the resulting clusters with the true hierarchical groupings. We use the adjusted rand index to measure clustering quality.

**Files used**: `clustering.py`, `dendrogram_generator.py`, `cluster_evaluation.py`

### Visualization and Interpretation

We employ various visualization techniques to interpret our results:
- PCA and t-SNE for dimensionality reduction
- Confusion matrices for classification results
- Dendrograms for hierarchical clustering

**Files used**: `visualization_tools.py`, `result_analysis.py`

Our experiments primarily focused on the "dog" hierarchy as a case study, but the methodology can be applied to any concept in WordNet to systematically evaluate how diffusion models encode hierarchical knowledge.
