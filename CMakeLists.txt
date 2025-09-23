# CMakeList.txt : Main CMake file for CracklingPlusPlus
cmake_minimum_required (VERSION 3.8)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Project name, version and description
project ("CracklingPlusPlus" LANGUAGES CXX C VERSION 1.0.0 DESCRIPTION "C++ version of Crackling" )

# Set CXX Standard for project
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Optimisation flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")

# Add library subdirectory
add_subdirectory (${CMAKE_SOURCE_DIR}/lib)

# Boost
find_package(Boost REQUIRED COMPONENTS regex iostreams)
include_directories("C:/local/boost_1_88_0")

# Add source to this project's executable.
add_executable (${PROJECT_NAME} ${CMAKE_SOURCE_DIR}/src/CracklingPlusPlus.cpp)

# Link Libraries to main executable
target_link_libraries (
	${PROJECT_NAME}
	PRIVATE Boost::dynamic_linking
	PRIVATE Boost::boost
	PRIVATE Boost::regex
	PRIVATE Boost::iostreams
	PRIVATE Logger 
	PRIVATE configParserModule
	PRIVATE cas9InputModule
	PRIVATE chopchopModule
	PRIVATE mm10dbModule
	PRIVATE sgrnascorer2Module
	PRIVATE bowtie2Module
	PRIVATE ISSLScoringModule
	PRIVATE ISSLScoringModuleMMF
	PRIVATE outputModule
)

# Added ISSLCreateIndex subdirectory
add_subdirectory (${CMAKE_SOURCE_DIR}/ISSLCreateIndex)

# Added ISSLScoreOfftargets subdirectory
add_subdirectory (${CMAKE_SOURCE_DIR}/ISSLScoreOfftargets)

# Added ISSLScoreOfftargetsMMF subdirectory
add_subdirectory (${CMAKE_SOURCE_DIR}/ISSLScoreOfftargetsMMF)

# Added ExtractOffTargets subdirectory
add_subdirectory (${CMAKE_SOURCE_DIR}/ExtractOfftargets)

# Add tests subdirectory
# add_subdirectory (${CMAKE_SOURCE_DIR}/tests)

# TODO: Sort out testing data in single location
# Copy testing data
file(COPY ${CMAKE_SOURCE_DIR}/tests/data/ DESTINATION data/)
