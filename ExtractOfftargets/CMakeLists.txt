# Boost
find_package(Boost REQUIRED COMPONENTS regex)

# OpenMP (make it optional)
find_package(OpenMP)

# Add ExtractOfftarget executable
add_executable (ExtractOfftargets ExtractOfftargets.cpp ExtractOfftargets.hpp)

if(OpenMP_CXX_FOUND)
    target_link_libraries(ExtractOfftargets PRIVATE Boost::boost Boost::regex OpenMP::OpenMP_CXX utils)
else()
    target_link_libraries(ExtractOfftargets PRIVATE Boost::boost Boost::regex utils)
endif()