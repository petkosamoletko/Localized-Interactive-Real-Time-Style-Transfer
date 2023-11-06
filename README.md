# Interactive Style Transfer GUI: A Sophisticated Approach

Welcome to the repository of the Interactive Style Transfer GUI, a system that pioneers the integration of neural style transfer with precise localization for a fully customizable artistic experience. Designed for those who seek to harness the power of AI in their creative pursuits, this interface is the culmination of extensive research and development in the field of digital artistry.

![Short Demo](GIF.gif)

## Prerequisites

Before embarking on your journey with our system, ensure you have the following tools ready:

- Python 3.9.6
- Git
- pip

## Installation Guide

Follow these steps to set up the GUI:

1. Clone the repository to your local machine.
2. Navigate to the project directory.
3. Install the dependencies with the following command:

```bash
pip install -r requirements.txt
```

## Trained Models

Our system incorporates two state-of-the-art models for an unparalleled style transfer experience:

- The **AdaIN** model, specially and privately trained for this project, is supplied via the `Adain.pth` file.
- The **SAM** model (Segment Anything from Meta AI) requires a separate download from the official SAM GitHub page. Please place the downloaded model file in the submission folder as `sam_vit_h_4b8939.pth`.

For SAM's model, visit:
https://github.com/facebookresearch/segment-anything#model-checkpoints

## Running the Graphical Interface Application

To launch the Interactive Style Transfer GUI, navigate to the application's directory and run:

```bash
python GUI.py
```

## Project Features

This project builds upon the AdaIN foundation and introduces localization to address arbitrary style transfer limitations. Key features include:

- **Style Strength**: Adjust the application intensity of the style on the desired content image segment.
- **Interpolation**: Blend multiple style images over parts of the content image, each with respective weight, impacting the final outcome.
- **Color Palette Preservation via WCT**: Preserve the original colors of your content image, ensuring the style's influence doesn't override the inherent color scheme.
- **Segment-specific Color Customization**: Change the color palette of individual segments post-style application for precise artistic control.

## Project Evaluation

This thesis project was awarded with a **80.5%** project grade.


We invite you to engage with our GUI and transform imagery through the lens of modern AI-driven style transfer.

