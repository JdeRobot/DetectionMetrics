from detectionmetrics.models.torch_model_utils import o3d_randlanet

# Default functions
preprocess = o3d_randlanet.preprocess
transform_input = o3d_randlanet.transform_input
update_probs = o3d_randlanet.update_probs
