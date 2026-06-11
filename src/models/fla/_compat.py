"""
Compatibility shim for the fla subtree vendored into VLANeXt.

The original code lived under `kairos.third_party.fla` and imported
`FLAGS_KAIROS_IS_METAX` from kairos' hardware-detection module. That flag only
selects a Metax-GPU code path; on standard NVIDIA hardware it is False. We
expose it here as a constant so the vendored fla files have no dependency on the
kairos package.
"""

import os

# Metax GPU is not used in VLANeXt; honor an env override just in case.
FLAGS_KAIROS_IS_METAX = os.environ.get("FLAGS_KAIROS_IS_METAX", "0") == "1"
