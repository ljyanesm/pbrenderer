cmake_minimum_required(VERSION 2.8)

PROJECT(pbrenderer)
 
# packages                                                                                                                                                                                                 
FIND_PACKAGE(Qt4 REQUIRED)
FIND_PACKAGE(CUDA REQUIRED)
INCLUDE(FindCUDA)
INCLUDE(${QT_USE_FILE})

SET(pbrenderer_SOURCES main.cpp LYSpatialHash.cpp LYShader.cpp LYScreenspaceRenderer.cpp LYOctree.cpp LYMesh.cpp LYHapticKeyboard.cpp LYHapticDevice.cpp LYDistanceFieldCollider.cpp LYCell.cpp LYCamera.cpp LYWorld.cpp LYTimer.cpp LYTexture.cpp)
SET(pbrenderer_HEADERS LYSpatialHash.h LYSpaceHandler.h LYShader.h LYScreenspaceRenderer.h LYOctree.h LYMesh.h LYHapticState.h LYHapticKeyboard.h LYHapticInterface.h LYHapticDevice.h LYDistanceFieldCollider.h LYCollisionHandler.h LYCell.h LYCamera.h defines.h LYWorld.h LYVertex.h LYTimer.h LYTexture.h)

SET(QT_USE_QTOPENGL TRUE)

#QT4_WRAP_UI(UISrcs MyWidget.ui)
#QT4_WRAP_CPP(MOCSrcs MyWidget.h)
#QT4_WRAP_CPP(pbrenderer_HEADERS_MOC ${pbrenderer_HEADERS})
#QT4_WRAP_UI(pbrenderer_FORMS_HEADERS ${pbrenderer_FORMS})
#QT4_ADD_RESOURCES(pbrenderer_RESOURCES_RCC ${pbrenderer_RESOURCES})

include_directories(${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR})
 
ADD_EXECUTABLE(pbrenderer ${pbrenderer_SOURCES} 
    ${pbrenderer_HEADERS})
#	 Agregar estas variables a la configuracion para incluir el build de Qt
#    ${pbrenderer_HEADERS_MOC} 
#    ${pbrenderer_FORMS_HEADERS} 
#    ${pbrenderer_RESOURCES_RCC})
#INCLUDE_DIRECTORIES(${CMAKE_CURRENT_BINARY_DIR})
TARGET_LINK_LIBRARIES(pbrenderer ${QT_QTMAIN_LIBRARY} ${QT_LIBRARIES})
 
IF(WIN32) # Check if we are on Windows
  if(MSVC) # Check if we are using the Visual Studio compiler
    set_target_properties(pbrenderer PROPERTIES LINK_FLAGS_RELEASE "/SUBSYSTEM:WINDOWS")
  elseif(CMAKE_COMPILER_IS_GNUCXX)
    # SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mwindows") # Not tested
  else()
    message(SEND_ERROR "You are using an unsupported Windows compiler! (Not MSVC or GCC)")
  endif()
elseif(UNIX)
  # Nothing special required
else()
  message(SEND_ERROR "You are on an unsupported platform! (Not Win32 or Unix)")
ENDIF()