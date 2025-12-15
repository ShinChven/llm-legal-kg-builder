#!/usr/bin/env python3
"""
Script to check the length of children in a YAML file for a specific act.
The act title is read from the 'core_act_title' field in the YAML.
"""

import argparse
import yaml
import sys
from pathlib import Path


def check_children_length(yaml_file_path):
    """
    Read a YAML file and check the length of children for the core act.

    Args:
        yaml_file_path (str): Path to the YAML file

    Returns:
        dict: Information about the children count
    """
    try:
        # Read the YAML file
        with open(yaml_file_path, 'r', encoding='utf-8') as file:
            data = yaml.safe_load(file)

        # Get the core act title
        core_act_title = data.get('core_act_title')
        if not core_act_title:
            return {
                'error': 'core_act_title not found in YAML file',
                'success': False
            }

        # Find the core act in the data
        core_act_data = data.get(core_act_title)
        if not core_act_data:
            return {
                'error': f'Act "{core_act_title}" not found in YAML data',
                'success': False,
                'core_act_title': core_act_title
            }

        # Get the children
        children = core_act_data.get('children', {})
        children_count = len(children)

        # Prepare result
        result = {
            'success': True,
            'core_act_title': core_act_title,
            'children_count': children_count,
            'children_list': list(children.keys()) if children else []
        }

        # Add additional metadata from the YAML
        metadata_fields = ['model_name', 'chunk_sizes', 'max_workers', 'generated_at', 'extraction_type']
        for field in metadata_fields:
            if field in data:
                result[field] = data[field]

        return result

    except FileNotFoundError:
        return {
            'error': f'File not found: {yaml_file_path}',
            'success': False
        }
    except yaml.YAMLError as e:
        return {
            'error': f'Error parsing YAML file: {str(e)}',
            'success': False
        }
    except Exception as e:
        return {
            'error': f'Unexpected error: {str(e)}',
            'success': False
        }


def main():
    """Main function to handle command line arguments and execute the check."""
    parser = argparse.ArgumentParser(
        description='Check the length of children in a YAML file for a specific act'
    )
    parser.add_argument(
        'yaml_file',
        help='Path to the YAML file to analyze'
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Show detailed output including children list'
    )

    args = parser.parse_args()

    # Check if file exists
    yaml_path = Path(args.yaml_file)
    if not yaml_path.exists():
        print(f"Error: File '{args.yaml_file}' does not exist.")
        sys.exit(1)

    # Analyze the YAML file
    result = check_children_length(args.yaml_file)

    if not result['success']:
        print(f"Error: {result['error']}")
        sys.exit(1)

    # Print results
    print(f"Core Act Title: {result['core_act_title']}")
    print(f"Children Count: {result['children_count']}")

    if args.verbose:
        print(f"\nMetadata:")
        metadata_fields = ['model_name', 'chunk_sizes', 'max_workers', 'generated_at', 'extraction_type']
        for field in metadata_fields:
            if field in result:
                print(f"  {field}: {result[field]}")

        if result['children_list']:
            print(f"\nChildren:")
            for i, child in enumerate(result['children_list'], 1):
                print(f"  {i}. {child}")
        else:
            print("\nNo children found.")


if __name__ == '__main__':
    main()
