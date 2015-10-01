# README #

This is a point-based graphic and haptic rendering tool for dense point-clouds developed using OpenGL and CUDA. The file format used by the application for the point-based models is PLY, as well as for the visual haptic proxy object.

### How do I get set up? ###

* Summary of set up

This application is programmed using the C++ language and most of the files necessary to compile with VS2012 are included. The libraries for the project are under the 'project libraries' folder and have a 'include' and 'lib' library under each folder.

* Configuration

There is a configuration file called app.cfg which contains a list of configurable attributes such as:

* Dependencies

The libraries used are:
Boost system and filesystem v1.58.
CUDA v7.0.
Assimp v3.0 (July 2012).
Config4star: Cpp version.
SDL v1.2.15.
OpenHaptics v3.1.
glew v1.9.0.
freeglut v2.8.1.

* How to run tests

Place the models you want to load using the application on a folder named 'models' at the same directory level as the executable.

* Deployment instructions

Along with the executable, the 'dll' files of the different libraries should be available in the PATH environment variable or included in the same folder of the executable. The models used to visualise the haptic proxy object should also be included along with the force vector model. A folder containing the graphic shaders named 'Shaders' and a folder with the models named 'Models' should be on the same directory as the executable.

The project files and folders list

* models/
* Shaders/
* app.cfg
* arrow.ply
* Assimp32.dll
* bbox.ply
* boost_filesystem-vc110-mt-1_58.dll
* boost_system-vc110-mt-1_58.dll
* freeglut.dll
* glew32.dll
* hd.dll
* hip.ply
* pbrenderer.exe
* PhantomIoLib42.dll
* proxy.ply
* surface.ply

### Contribution guidelines ###

* Writing tests

* Code review

* Other guidelines

### Who do I talk to? ###

* Repo owner

Luis Yanes <yanes<dot>luis@gmail.com>

* Other community or team contact

Stephen Laycock, UEA, UK.