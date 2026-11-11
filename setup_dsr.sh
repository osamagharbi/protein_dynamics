#!/bin/bash
# Clone the upstream DSR code next to MAMBA-DSR/ and diffusion_dsr/ (they import DSR/lib via ../DSR)
# and apply the local numpy>=1.24 / scikit-image>=0.19 compatibility fixes + extra configs.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
[ -d DSR/.git ] || git clone https://github.com/Sundw-818/DSR.git DSR
git -C DSR checkout -q 6350f24249a7931d76e415698081c295ff1de9c9
git -C DSR apply --check ../patches/dsr_numpy_skimage_compat.patch 2>/dev/null \
  && git -C DSR apply ../patches/dsr_numpy_skimage_compat.patch && echo "patch applied" || echo "patch already applied"
for f in patches/dsr_confs/*.conf; do [ -e "DSR/confs/${f##*/}" ] || cp "$f" DSR/confs/; done
echo "DSR ready in $(pwd)/DSR (build lib/inside_mesh with: cd DSR/lib/inside_mesh && python setup.py build_ext --inplace)"
