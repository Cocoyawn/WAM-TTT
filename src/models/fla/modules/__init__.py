# Trimmed for VLANeXt ablation: the original aggregator imported many modules
# (fused_bitlinear, fused_*_cross_entropy, mlp, ...) that pull in torch>=2.4
# tensor-parallel APIs (DeviceMesh) unused by GatedDeltaNet. Consumers import
# the specific submodules they need directly, so this package init is empty.
