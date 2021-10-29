cmake_minimum_required(VERSION 3.15)

project(dpg)

# Boost
find_package(Boost REQUIRED COMPONENTS timer chrono system program_options)

# OpenMP
find_package(OpenMP REQUIRED COMPONENTS CXX)

add_library(${PROJECT_NAME} STATIC)

target_sources(${PROJECT_NAME} PRIVATE src/kgraph-data.h src/kgraph.cpp src/metric.cpp src/RandGen.cpp src/RandGen.h)

target_include_directories(${PROJECT_NAME} PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/src>
                                                  $<INSTALL_INTERFACE:include>)

target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_17)

target_link_libraries(${PROJECT_NAME} PUBLIC OpenMP::OpenMP_CXX Boost::headers Boost::program_options Boost::timer
                                             Boost::chrono Boost::system)

target_compile_options(${PROJECT_NAME} PUBLIC $<$<CXX_COMPILER_ID:MSVC>:/arch:AVX2>
                                              $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-march=native>)

function(dpg_add_executable _name)
    add_executable(${_name})

    target_sources(${_name} PRIVATE src/${_name}.cpp)

    target_link_libraries(${_name} PRIVATE ${PROJECT_NAME})
endfunction()

dpg_add_executable(DPG_diverse)
dpg_add_executable(DPG_index)
dpg_add_executable(DPG_search)

dpg_add_executable(DPG_diverse_sift1m)
dpg_add_executable(DPG_diverse_sift1m_evaluation)
dpg_add_executable(DPG_diverse_sift1m_stats)
dpg_add_executable(DPG_index_sift1m)
