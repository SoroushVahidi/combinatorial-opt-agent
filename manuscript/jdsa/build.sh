#!/bin/sh
set -eu
package_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
OUTPUT_DIR="$package_dir" sh "$package_dir/source/build.sh"
