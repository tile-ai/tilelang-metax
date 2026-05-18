# MACA backend: toolchain, and source files
if (NOT USE_MACA)
  return()
endif()

include(cmake/FindMACA.cmake)
find_maca(${USE_MACA})

file(GLOB TILE_LANG_MACA_SRCS
  src/target/codegen_utils.cc
  src/backend/maca/op/*.cc
  src/backend/maca/codegen/codegen_maca.cc
  src/backend/maca/codegen/rt_mod_maca.cc
  src/backend/maca/runtime/maca_module.cc
  src/backend/maca/runtime/maca_target_kind.cc
  src/backend/maca/runtime/maca_device_api.cc
)
list(APPEND TILE_LANG_SRCS ${TILE_LANG_MACA_SRCS})

list(APPEND TILE_LANG_INCLUDES ${MACA_INCLUDE_DIRS})
