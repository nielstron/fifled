cmake_minimum_required (VERSION 2.6)
project (fifled)

# Define output
file(GLOB SOURCES "src/*.cpp")
add_executable(fifled ${SOURCES}) 

# OpenCV
find_package( OpenCV REQUIRED )
target_link_libraries(fifled ${OpenCV_LIBS})

add_compile_options("-Ofast")