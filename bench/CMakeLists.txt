cmake_policy(SET CMP0015 NEW)

aux_source_directory(. SRC_LIST) # find add source files in the directory . , named SRC_LIST

include_directories(BEFORE ..) # search header files BEFORE/AFTER to specify prepend/append

set(EXECUTABLE ethminer)

# GLOB will generate a list of all files that match the globbing expressions and store it into the variable. 
# Globbing expressions are similar to regular expressions, but much simpler.
file(GLOB HEADERS "*.h")

add_executable(${EXECUTABLE} ${SRC_LIST} ${HEADERS})

# Make a top-level <target> depend on other top-level targets to ensure that they build before <target> does
# A top-level target is one created by ADD_EXECUTABLE, ADD_LIBRARY, or ADD_CUSTOM_TARGET.
add_dependencies(${EXECUTABLE} BuildInfo.h) 

target_link_libraries(${EXECUTABLE} ethcore)
target_link_libraries(${EXECUTABLE} ethash)
target_link_libraries(${EXECUTABLE} devcore libjson-rpc-cpp::client)

if(ETHDBUS)
	find_package(PkgConfig)
	set( ENV{PKG_CONFIG_PATH} "$ENV{PKG_CONFIG_PATH}:/usr/lib/x86_64-linux-gnu/pkgconfig" )
	pkg_check_modules(DBUS dbus-1)
	include_directories(${DBUS_INCLUDE_DIRS})
	link_directories(${DBUS_LIBRARY_DIRS})
	target_link_libraries(${EXECUTABLE} ${DBUS_LIBRARIES})
endif()

if(APICORE)
	target_link_libraries(${EXECUTABLE} apicore)
endif()

if(ETHSTRATUM)
    target_link_libraries(${EXECUTABLE} ethstratum)
endif()

include(GNUInstallDirs)
install(TARGETS ethminer DESTINATION ${CMAKE_INSTALL_BINDIR})
if(MSVC)
    install(FILES $<TARGET_PDB_FILE:ethminer> DESTINATION ${CMAKE_INSTALL_BINDIR} OPTIONAL)
endif()
