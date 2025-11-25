OS environment: macOS

Run the following commands to set up project:
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

Run the following command to run the base detector implementation:
python3 base_detector.py

Run the following command to run the feature augmentented detector implementation:
python3 feature_augmented_detector.py

Run the following command to run the group-level detector:
python3 group_level_detector.py

Run the following command to run the rating-scale-analysis:
python3 rating_scale_analysis_detector.py

Run the following command to create the group size graph:
python3 group_size_graph.py

Run the following command to create the judgement dimension graph:
python3 judgement_dimension_graph.py

Run the following command to create the judgement dimension graph:
python3 rating_scale_graph.py