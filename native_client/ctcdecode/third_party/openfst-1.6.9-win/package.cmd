@echo off
setlocal enableextensions

:: Locate winrar.exe
set winrar="%ProgramFiles%\winrar\winrar.exe"
if not exist %winrar% set winrar="%ProgramFiles(x86)%\winrar\winrar.exe"
if not exist %winrar% (
  echo ERROR: Cannot find winrar.exe 1>&2
  exit /b 1
)

:: Get head tag and make sure it is sane.
for /f %%i in ('git describe --long --dirty') do set longtag=%%i
set insane=0
if not %longtag:-dirty=%.==%longtag%. set insane=1
if %longtag:-0-g=%.==%longtag%.       set insane=1
if %longtag:win/=%.==%longtag%.       set insane=1
if %insane%==1 (
    echo ERROR: Best HEAD description '%longtag%' is not at a clean tag on winport branch 1>&2
    exit /b 1
)

:: Get short description now, strip "win/" for version only
for /f %%i in ('git describe') do set tag=%%i
set tag=%tag:win/=%

:: Create archive with a comment
set cmtfile=%TEMP%\openfst-package-comment.txt
del /q %cmtfile% 2>nul
echo OpenFST binaries for Windows x64, optimized build.  >>%cmtfile%
echo Copyright 2005-2018 Google, Inc. (Original source). >>%cmtfile%
echo Copyright 2016-2018 SmartAction LLC (Windows port). >>%cmtfile%
echo Copyright 2016-2018 Johns Hopkins Uni (Windows port). >>%cmtfile%
echo.>>%cmtfile%
echo OpenFST home page: http://www.openfst.org/         >>%cmtfile%
echo Git Repository: https://github.com/kkm000/openfst/ >>%cmtfile%
echo Build tag: %longtag%                               >>%cmtfile%

set zipfile=openfst-bin-win-x64-%tag%.zip
del /q %zipfile% 2>nul
%winrar% a -ep -m5 -z%cmtfile% %zipfile% NEWS COPYING build_output\x64\Release\bin\*.exe
if errorlevel 1 (
  echo "ERROR: Cannot create archive '%zipfile%' 1>&2
  del /q %cmtfile% 2>nul
  exit /b 1
)

echo SUCCESS: Created archive '%zipfile%' 1>&2
del /q %cmtfile% 2>nul
exit /b 0
