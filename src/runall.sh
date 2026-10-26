echo "==========================================="
echo "                                           "
echo "                                           "
echo "                  FULL-LUT                 "
echo "                                           "
echo "                                           "
echo "==========================================="

full_lut() {
    LUT_MODE="full" PARAMS_FILE=$1 cargo bench --bench $2 --features "avx, mem-profile" --config 'build.rustflags=["-C","target-feature=+avx2,+fma"]'
    LUT_MODE="full" PARAMS_FILE=$1 cargo bench --bench $2 --features "avx" --config 'build.rustflags=["-C","target-feature=+avx2,+fma"]'
}

full_lut "../paramux/example_finder/vertical_packing/depth_2/s1.json" vpack
full_lut "../paramux/example_finder/vertical_packing/depth_2/s3.json" vpack
full_lut "../paramux/example_finder/ohvpack/depth_2/s1.json" ohvpack
full_lut "../paramux/example_finder/ohvpack/depth_2/s3.json" ohvpack


echo "==========================================="
echo "                                           "
echo "                                           "
echo "                BENCHMARKS                 "
echo "                 (STREAM)                  "
echo "                                           "
echo "                                           "
echo "==========================================="

# Benchmarks
./bench.sh "../paramux/example_finder/vertical_packing/depth_2/s1.json" vpack
./bench.sh "../paramux/example_finder/vertical_packing/depth_2/s3.json" vpack
./bench.sh "../paramux/example_finder/ohvpack/depth_2/s1.json" ohvpack
./bench.sh "../paramux/example_finder/ohvpack/depth_2/s3.json" ohvpack

./bench.sh "../paramux/example_finder/vertical_packing/depth_3/s1.json" vpack
./bench.sh "../paramux/example_finder/vertical_packing/depth_3/s2.json" vpack
./bench.sh "../paramux/example_finder/vertical_packing/depth_3/s3.json" vpack
./bench.sh "../paramux/example_finder/ohvpack/depth_3/s1.json" ohvpack
./bench.sh "../paramux/example_finder/ohvpack/depth_3/s3.json" ohvpack
