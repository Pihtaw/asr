cmake_minimum_required(VERSION 3.10)
project(ASR_System)

# Устанавливаем стандарт C++17 и флаги оптимизации
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

# Поиск необходимых библиотек
find_package(PkgConfig REQUIRED)
pkg_check_modules(SNDFILE REQUIRED sndfile)

# Устанавливаем выходные директории
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR}/lib)

# Структура проекта
set(SOURCE_DIR src)
set(INCLUDE_DIR include)
set(TEST_DIR test)

# Список исходных файлов
set(SOURCES
    ${SOURCE_DIR}/audio.cpp
    ${SOURCE_DIR}/cnn.cpp
    ${SOURCE_DIR}/lstm.cpp
    ${SOURCE_DIR}/ctc.cpp
    ${SOURCE_DIR}/asr_model.cpp
    ${SOURCE_DIR}/main.cpp
    ${SOURCE_DIR}/libre.cpp
)

# Включаем директории
include_directories(
    ${INCLUDE_DIR}
    ${SNDFILE_INCLUDE_DIRS}
)

# Основной исполняемый файл
add_executable(asr ${SOURCES})

# Линковка библиотек
target_link_libraries(asr
    PRIVATE
    m
    ${SNDFILE_LIBRARIES}
)

# Установка для make install
install(TARGETS asr DESTINATION bin)
install(DIRECTORY ${INCLUDE_DIR}/ DESTINATION include)

# Тестирование (если включено)
option(BUILD_TESTS "Build tests" OFF)
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(${TEST_DIR})
endif()

# Копирование тестовых данных (правильный способ)
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/data)
file(COPY ${CMAKE_SOURCE_DIR}/data/ DESTINATION ${CMAKE_BINARY_DIR}/data)

# Дополнительные цели
add_custom_command(TARGET asr POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E echo "=== Running compiled executable ==="
    COMMAND $<TARGET_FILE:asr>
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Running the ASR system"
)

# Информация о конфигурации
message(STATUS "Libsndfile include dirs: ${SNDFILE_INCLUDE_DIRS}")
message(STATUS "Libsndfile libraries: ${SNDFILE_LIBRARIES}")
message(STATUS "Executable will be built to: ${EXECUTABLE_OUTPUT_PATH}")
message(STATUS "Source files: ${SOURCES}")

# Проверка существования исходных файлов
foreach(src_file ${SOURCES})
    if(NOT EXISTS ${CMAKE_SOURCE_DIR}/${src_file})
        message(WARNING "Missing source file: ${src_file}")
    endif()
endforeach()