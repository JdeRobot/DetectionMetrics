file(REMOVE_RECURSE
  "libDeepLearningSuite.pdb"
  "libDeepLearningSuite.so"
)

# Per-language clean rules from dependency scanning.
foreach(lang CXX)
  include(CMakeFiles/DeepLearningSuite.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
