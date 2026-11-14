cmake_minimum_required(VERSION 3.20)
project(DFTLSDA VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

include(CheckCXXCompilerFlag)
check_cxx_compiler_flag(-mcpu=native HAS_MCPU_NATIVE)
check_cxx_compiler_flag(-march=native HAS_MARCH_NATIVE)
if(HAS_MCPU_NATIVE)
  set(NATIVE_FLAG -mcpu=native)
elseif(HAS_MARCH_NATIVE)
  set(NATIVE_FLAG -march=native)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O3 ${NATIVE_FLAG}")

# --- dependencies -----------------------------------------------------------
find_package(OpenMP)
find_package(LAPACK REQUIRED)        # Accelerate / OpenBLAS / MKL
find_path(FFTW3_INCLUDE_DIR fftw3.h HINTS /opt/homebrew/include /usr/local/include ENV FFTW_ROOT PATH_SUFFIXES include)
find_library(FFTW3_LIBRARY NAMES fftw3 HINTS /opt/homebrew/lib /usr/local/lib ENV FFTW_ROOT PATH_SUFFIXES lib)
if(NOT FFTW3_INCLUDE_DIR OR NOT FFTW3_LIBRARY)
  message(FATAL_ERROR "FFTW3 not found (set FFTW_ROOT)")
endif()

# --- core library -----------------------------------------------------------
add_library(lsdacore STATIC
  src/base/common.cpp
  src/base/linalg.cpp
  src/base/fft.cpp
  src/math/special.cpp
  src/math/radial.cpp
  src/xc/lda.cpp
  src/xc/pbe.cpp
  src/paw/xml.cpp
  src/paw/setup.cpp
  src/crystal/crystal.cpp
  src/crystal/symmetry.cpp
  src/dft/basis.cpp
  src/dft/hamiltonian.cpp
  src/dft/eigensolver.cpp
  src/dft/occupations.cpp
  src/dft/mixer.cpp
  src/dft/calculation.cpp
  src/dft/forces.cpp
  src/dft/relax.cpp
  src/dft/neb.cpp
  src/dft/phonons.cpp
  src/io/input.cpp
)
target_include_directories(lsdacore PUBLIC src ${FFTW3_INCLUDE_DIR})
target_link_libraries(lsdacore PUBLIC ${FFTW3_LIBRARY} ${LAPACK_LIBRARIES})
if(OpenMP_CXX_FOUND)
  target_link_libraries(lsdacore PUBLIC OpenMP::OpenMP_CXX)
else()
  message(WARNING "OpenMP not found: building serial code")
endif()
target_compile_options(lsdacore PRIVATE -Wall -Wextra -Wno-unused-parameter)

add_executable(lsda src/main.cpp)
target_link_libraries(lsda PRIVATE lsdacore)

# --- tests ------------------------------------------------------------------
enable_testing()
add_executable(test_unit tests/test_unit.cpp)
target_link_libraries(test_unit PRIVATE lsdacore)
add_test(NAME unit COMMAND test_unit ${CMAKE_SOURCE_DIR}/pseudos)
find_package(Python3 COMPONENTS Interpreter)
if(Python3_FOUND)
  add_test(NAME regression
           COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tests/run_regression.py $<TARGET_FILE:lsda> ${CMAKE_SOURCE_DIR}/pseudos)
endif()
