#!/bin/sh
SRC="platform_osx.mm ao.cc tiny_obj_loader.cc"
LIBS="-framework Cocoa -framework CoreVideo -framework OpenGL"
g++ -std=c++11 $SRC $LIBS -o ao

