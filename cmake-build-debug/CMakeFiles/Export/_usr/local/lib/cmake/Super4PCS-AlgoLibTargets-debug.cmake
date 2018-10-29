#----------------------------------------------------------------
# Generated CMake target import file for configuration "Debug".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "super4pcs_algo" for configuration "Debug"
set_property(TARGET super4pcs_algo APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(super4pcs_algo PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_DEBUG "CXX"
  IMPORTED_LOCATION_DEBUG "${_IMPORT_PREFIX}/lib/libsuper4pcs_algo.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS super4pcs_algo )
list(APPEND _IMPORT_CHECK_FILES_FOR_super4pcs_algo "${_IMPORT_PREFIX}/lib/libsuper4pcs_algo.a" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
