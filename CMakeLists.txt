cmake_minimum_required (VERSION 2.6)
project (ImageFlow)

# Define output
file(GLOB SOURCES "src/*.cpp")
add_executable(imageflow ${SOURCES}) 

# OpenCV
find_package( OpenCV REQUIRED )
target_link_libraries(imageflow ${OpenCV_LIBS})

add_compile_options("-Ofast")