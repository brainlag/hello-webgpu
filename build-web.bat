@echo off
rem Builds the Emscripten implementation (on Windows)
rem TODO: CMake...
rem 

if "%~1"=="/d" (
  set DEBUG=true
) else (
  set DEBUG=false
)

set CPP_FLAGS=-std=c++11 -Wall -Wextra -Wno-nonportable-include-path -fno-exceptions -fno-rtti
set EMS_FLAGS=-s LLD_REPORT_UNDEFINED -s ERROR_ON_UNDEFINED_SYMBOLS=0 -s ALLOW_MEMORY_GROWTH=0 -s ENVIRONMENT=web  -s NO_EXIT_RUNTIME=1 -s NO_FILESYSTEM=1 -s STRICT=1 -s TEXTDECODER=2 -s USE_WEBGPU=1 -s WASM=1 -s USE_SDL=2 -s ASYNCIFY
set OPT_FLAGS=

if %DEBUG%==true (
  set CPP_FLAGS=%CPP_FLAGS% -g3 -D_DEBUG=1 -Wno-unused 
  set EMS_FLAGS=%EMS_FLAGS%  -s ASSERTIONS=2 -s DEMANGLE_SUPPORT=1 -s SAFE_HEAP=1 -s STACK_OVERFLOW_CHECK=2 -s DISABLE_EXCEPTION_CATCHING=1 -s SUPPORT_ERRNO=0 
  set OPT_FLAGS=%OPT_FLAGS% -O0
) else (
  set CPP_FLAGS=%CPP_FLAGS% -g0 -DNDEBUG=1 -flto
  rem set CPP_FLAGS=%CPP_FLAGS% -g -fdebug-compilation-dir='../../..'
  set EMS_FLAGS=%EMS_FLAGS% -s ABORTING_MALLOC=0 -s ASSERTIONS=0 -s DISABLE_EXCEPTION_CATCHING=1  -s SUPPORT_ERRNO=0 --shell-file src/ems/shell.html
  set OPT_FLAGS=%OPT_FLAGS% -O3 
)

set SRC=
for %%f in (src/ems/*.cpp) do call set SRC=%%SRC%%src/ems/%%f 
for %%f in (src/*.cpp) do call set SRC=%%SRC%%src/%%f 
set SRC=%SRC%lib/imgui/imgui.cpp 
set SRC=%SRC%lib/imgui/imgui_demo.cpp 
set SRC=%SRC%lib/imgui/imgui_draw.cpp 
set SRC=%SRC%lib/imgui/imgui_tables.cpp 
set SRC=%SRC%lib/imgui/imgui_widgets.cpp 
set SRC=%SRC%lib/imgui/backends/imgui_impl_wgpu.cpp 
set SRC=%SRC%lib/imgui/backends/imgui_impl_sdl.cpp 

set INC=-Iinc -Ilib/imgui -Ilib/imgui/backends -I%USERPROFILE%/glm

set OUT=index
if not exist out\web mkdir out\web

rem Grab the Binaryen path from the ".emscripten" file (which needs to have
rem been set). We then swap the Unix-style slashes.
rem 
for /f "tokens=*" %%t in ('em-config BINARYEN_ROOT') do (set BINARYEN_ROOT=%%t)
set "BINARYEN_ROOT=%BINARYEN_ROOT:/=\%"

echo "emcc %CPP_FLAGS% %OPT_FLAGS% %EMS_FLAGS% %INC% %SRC% -o %OUT%.html"

%SystemRoot%\system32\cmd /c "em++ %CPP_FLAGS% %OPT_FLAGS% %EMS_FLAGS% %INC% %SRC% -o %OUT%.html"
set EMCC_ERR=%errorlevel%
if %DEBUG%==false (
  if %EMCC_ERR%==0 (
    %SystemRoot%\system32\cmd /c "%BINARYEN_ROOT%\bin\wasm-opt %OPT_FLAGS% --converge %OUT%.wasm -o %OUT%.wasm"
    set EMCC_ERR=%errorlevel%
  )
)

if %EMCC_ERR%==0 (
  echo Success!
)
exit /b %EMCC_ERR%
