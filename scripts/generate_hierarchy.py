#!/usr/bin/env python
"""
Command-line script to extract WordNet hierarchies and save them as CSV files.

Usage:
    python generate_hierarchy.py dog --output data/synset_hierarchies/dog.csv
"""
import os
import sys
import argparse
import matplotlib.pyplot as plt

# Add the parent directory to the path to import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.wordnet_hierarchy_extractor import extract_hierarchy
from src.visualization_tools import visualize_hierarchy

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Extract a hierarchy from WordNet and save it as a CSV file"
    )
    
    # Required arguments
    parser.add_argument(
        "concept", 
        help="The root concept to start extraction from (e.g., 'dog', 'vehicle')"
    )
    
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Output file path for the CSV (required)"
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
        "--visualize", "-v",
        action="store_true",
        help="Generate visualizations of the hierarchy"
    )
    
    parser.add_argument(
        "--vis-dir",
        help="Directory to save visualizations (required if --visualize is used)"
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
    
    output_file = args.output
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    if args.visualize and not args.vis_dir:
        parser.error("--vis-dir is required if --visualize is used")
    
    # Extract the hierarchy
    print(f"Extracting hierarchy for '{args.concept}'...")
    print(f"  Max depth: {args.max_depth}")
    print(f"  Frequency threshold: {args.freq_threshold}")
    print(f"  Output file: {output_file}")
    
    try:
        hier_df, graph = extract_hierarchy(
            args.concept,
            max_depth=args.max_depth,
            frequency_threshold=args.freq_threshold,
            output_file=output_file
        )
        
        print(f"Successfully created hierarchy with {len(graph.nodes)} nodes")
        print(f"Saved to {output_file}")
        
        # Generate visualizations if requested
        if args.visualize:
            if not args.vis_dir:
                # Default to same directory as output file if not specified
                vis_dir = os.path.dirname(output_file)
                print("No visualization directory specified, using output directory")
            else:
                vis_dir = args.vis_dir
            
            os.makedirs(vis_dir, exist_ok=True)
            print(f"Generating visualizations in {vis_dir}...")
            
            for layout in args.layouts:
                concept_name = os.path.splitext(os.path.basename(output_file))[0]
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
