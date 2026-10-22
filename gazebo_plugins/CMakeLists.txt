# RobotLab Gazebo system plugins. Build with `robotlab-build-plugins`
# (or: cmake -S gazebo_plugins -B build/gazebo_plugins && cmake --build build/gazebo_plugins).
# Needs the Gazebo Harmonic development files (ROS Jazzy vendor packages here).
cmake_minimum_required(VERSION 3.22)
project(robotlab_gazebo_plugins LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(gz-sim8 REQUIRED)
find_package(gz-plugin2 REQUIRED COMPONENTS register)
find_package(gz-transport13 REQUIRED)

add_library(robotlab-lockstep SHARED lockstep/RobotlabLockstep.cc)
target_link_libraries(robotlab-lockstep
  PRIVATE gz-sim8::gz-sim8 gz-plugin2::register gz-transport13::gz-transport13)
