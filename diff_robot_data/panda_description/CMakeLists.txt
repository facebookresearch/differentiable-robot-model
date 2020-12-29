cmake_minimum_required(VERSION 3.1.3)
project(moveit_resources_panda_description)
find_package(catkin REQUIRED)

catkin_package()

install(DIRECTORY meshes urdf DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})
