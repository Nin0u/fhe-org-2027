echo "====== NOISE  PROFILING ====="
PARAMS_FILE=$1 cargo bench --bench $2 --features "avx, noise-profile" --config 'build.rustflags=["-C","target-feature=+avx2,+fma"]'
echo "============================="
echo
echo "====== MEMORY PROFILING ====="
PARAMS_FILE=$1 cargo bench --bench $2 --features "avx, mem-profile" --config 'build.rustflags=["-C","target-feature=+avx2,+fma"]'
echo "============================="
echo
echo "======  TIME  PROFILING ====="
PARAMS_FILE=$1 cargo bench --bench $2 --features "avx, time-profile" --config 'build.rustflags=["-C","target-feature=+avx2,+fma"]'
echo "============================="
echo
echo "======  EXEC TIME BENCH ====="
PARAMS_FILE=$1 cargo bench --bench $2 --features "avx" --config 'build.rustflags=["-C","target-feature=+avx2,+fma"]'
echo "============================="
