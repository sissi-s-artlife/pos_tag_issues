# Part-of-Speech Tagging Visualization

This Python script performs Part-of-Speech (POS) tagging on a given text and visualizes the distribution of POS tags using Matplotlib. It also displays the word-POS tag correspondence after the visualization.

## Dependencies

- Python 3.x
- Matplotlib
- NLTK
- CRF Tagger (python-crfsuite)

## Installation

1. Ensure you have Python 3.x installed.
2. Install the required libraries:
    ```bash
    pip install matplotlib nltk python-crfsuite
    ```
3. Download NLTK's 'treebank' dataset:
    ```python
    import nltk
    nltk.download('treebank')
    ```

## Usage

1. Run the script `pos_tag_visualization.py`.
    ```bash
    python pos_tag_visualization.py
    ```

2. The script will:
    - Perform POS tagging on a predefined text.
    - Display a bar chart representing the distribution of POS tags.
    - Print the word-POS tag correspondence in the terminal/console after closing the plot window.

## Why Use This Code

- Gain insights into the distribution of POS tags in a text.
- Visualize the frequency of different POS tags, aiding in linguistic analysis and understanding.

