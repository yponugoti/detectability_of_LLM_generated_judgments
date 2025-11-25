# On the Detectability of LLM-generated Judgments

**OS environment:** macOS

## Setup

Run the following commands to set up project:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

### Base Detector Implementation

Run the following command to run the base detector implementation:

```bash
python3 base_detector.py
```

### Feature Augmented Detector Implementation

Run the following command to run the feature augmentented detector implementation:

```bash
python3 feature_augmented_detector.py
```

### Group-Level Detector

Run the following command to run the group-level detector:

```bash
python3 group_level_detector.py
```

### Rating Scale Analysis

Run the following command to run the rating-scale-analysis:

```bash
python3 rating_scale_analysis_detector.py
```

## Visualization

### Group Size Graph

Run the following command to create the group size graph:

```bash
python3 group_size_graph.py
```

### Judgement Dimension Graph

Run the following command to create the judgement dimension graph:

```bash
python3 judgement_dimension_graph.py
```

### Rating Scale Graph

Run the following command to create the judgement dimension graph:

```bash
python3 rating_scale_graph.py
```
