# Sourced by the tools/gri scripts. GRI_CHARS: GRI_Dan's character output root; BLENDER: blender executable.
GRI_CHARS=${GRI_CHARS:-/c/Repos/GRI_Dan/tools/out/characters}
BLENDER=${BLENDER:-"/c/Program Files/Blender Foundation/Blender 5.2/blender.exe"}
H=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$H/../.." && pwd)
PY=$ROOT/.venv/Scripts/python
cd "$ROOT"
