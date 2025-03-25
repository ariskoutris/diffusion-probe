#!/usr/bin/env python
"""
Command-line script to extract WordNet hierarchies and save them as JSON files.

Usage:
    python generate_hierarchy.py dog --output data/synset_hierarchies/dog.json
"""
import os
import sys
import argparse
import matplotlib.pyplot as plt
import pandas as pd
import json
from pathlib import Path

# Add the parent directory to the path to import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.wordnet_hierarchy_extractor import extract_hierarchy
from utils.visualization_tools import visualize_hierarchy

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Extract a hierarchy from WordNet and save it as a JSON file"
    )
    
    # Required arguments
    parser.add_argument(
        "concept", 
        help="The root concept to start extraction from (e.g., 'dog', 'vehicle')"
    )
    
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Output file path for the JSON file"
    )
    
    # Optional arguments
    parser.add_argument(
        "--max-depth", "-d",
        type=int, 
        default=2,
        help="Maximum depth to traverse in the hierarchy (default: 2)"
    )
    
    parser.add_argument(
        "--freq-threshold", "-f",
        type=int, 
        default=0,
        help="Minimum frequency threshold for including a synset (default: 0)"
    )
    
    parser.add_argument(
        "--vis-dir",
        help="Directory to save visualizations (if provided, visualizations will be generated)"
    )
    
    parser.add_argument(
        "--layouts",
        nargs="+",
        default=["dot", "twopi"],
        choices=["dot", "twopi", "neato", "circo", "fdp", "spring"],
        help="Layout algorithms to use for visualizations (default: dot twopi)"
    )
    
    parser.add_argument(
        "--use-pos",
        help="Specify part of speech (e.g., 'n' for noun, 'v' for verb)"
    )
    
    parser.add_argument(
        "--index",
        type=int,
        default=0,
        help="Index of synset to use when multiple are available (default: 0)"
    )

    return parser

def main():
    """Main function to extract and save hierarchies."""
    parser = parse_arguments()
    args = parser.parse_args()
    
    # Ensure output ends with .json
    output_path = args.output
    if not output_path.endswith('.json'):
        output_path = f"{output_path}.json"
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Extract the hierarchy
    print(f"Extracting hierarchy for '{args.concept}'...")
    print(f"  Max depth: {args.max_depth}")
    print(f"  Frequency threshold: {args.freq_threshold}")
    print(f"  Output file: {output_path}")
    
    try:
        hier_df, graph = extract_hierarchy(
            args.concept,
            max_depth=args.max_depth,
            frequency_threshold=args.freq_threshold,
            output_file=None  # Don't save CSV
        )
        
        print(f"Successfully created hierarchy with {len(graph.nodes)} nodes")
        
        # Create hierarchy data for JSON
        simplified_data = []
        
        # Find depth columns to extract paths
        depth_cols = [col for col in hier_df.columns if col.startswith('cat_depth_')]
        depth_cols.sort()
        
        # Process each row to create path information
        for _, row in hier_df.iterrows():
            # Get the formatted path (remove None values)
            path_names = [row[col] for col in depth_cols if pd.notna(row[col])]
            
            # Create entry with only essential data
            simplified_data.append({
                'synset_id': row['synset_id'],
                'name': row['class'],
                'definition': row['definition'],
                'path_names': path_names
            })
        
        # Save data to JSON
        with open(output_path, 'w') as f:
            json.dump(simplified_data, f, indent=2)
        print(f"Saved hierarchy to {output_path}")
        
        # Generate visualizations if vis_dir is provided
        if args.vis_dir:
            vis_dir = args.vis_dir
            os.makedirs(vis_dir, exist_ok=True)
            print(f"Generating visualizations in {vis_dir}...")
            
            for layout in args.layouts:
                concept_name = os.path.basename(output_path).replace('.json', '')
                vis_file = os.path.join(vis_dir, f"{concept_name}_{layout}.png")
                print(f"  Creating {layout} layout...")
                
                try:
                    fig, ax = visualize_hierarchy(
                        graph,
                        layout=layout,
                        figsize=(12, 10),
                        font_size=7,
                        show_bbox=True
                    )
                    plt.savefig(vis_file)
                    plt.close(fig)
                    print(f"  Saved to {vis_file}")
                except Exception as e:
                    print(f"  Error creating {layout} layout: {str(e)}")
    
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
