#!/bin/sh

#You, the user, should set this if it differes from what is set here!
JAVA_HOME=/usr/lib/jvm/java-1.11.0-openjdk-amd64

#Colours used in the shell
RED='\033[0;31m'
NC='\033[0m' # No Color
GREEN='\033[0;32m'

#Function is called whenever a program fails, $1 is supposed to be a short name for what failed (e.g Make)
fail() {
    printf "${RED}FAIL${NC}\n"
    echo "${RED}Setup failed: $1 $NC "
    exit 1
}

printf "Performing checks..."

#Check if libdeepspeech.so exists
if [ ! -e ./libdeepspeech/libs/libdeepspeech.so ]
then
    printf "${RED}FAIL${NC}\n"
    echo "- ./libdeepspeech/libs/libdeepspeech.so: ${RED}FAIL${NC}"
    echo "Download libdeepspeech.so from GitHub releases or compile DeepSpeech first"
    exit 1
fi

#Check if JAVA_HOME is set
if [ -z "$JAVA_HOME" ]
then
    printf "${RED}FAIL${NC}\n"
    echo "- JAVA_HOME: ${RED}FAIL${NC}"
    echo "Please set the JAVA_HOME environmental variable!"
    exit 1
fi

#Check if the gradle wrapper is present
if [ ! -d "./gradle" ]
then
    printf "${RED}FAIL${NC}\n"
    echo "- Gradle Wrapper: ${RED}FAIL${NC}"
    echo "Use 'gradle wrapper' and try again"
    exit 1
fi

#Check if the script is run as root, this is needed because Make
if [ $(id -u) -ne 0 ]
then
    printf "${RED}FAIL${NC}\n"
    echo "- Root check: ${RED}FAIL${NC}"
    echo "Please run this script as root or with sudo!"
    exit 1
fi

#Checks are complete
printf "${GREEN}OK${NC}\n"

#Create the .cpp file for the bindings
printf "Creating bindings..."
swig -c++ -java -package org.deepspeech.libdeepspeech -outdir libdeepspeech/src/main/java/org/deepspeech/libdeepspeech/ -o jni/deepspeech_wrap.cpp jni/deepspeech.i >/dev/null || fail "Swig"
printf "${GREEN}OK${NC}\n"

printf "Compiling libdeepspeech-jni.so..."

#We move into the libdeepspeech directory, to avoid funky behaviour
cd libdeepspeech

#We need to set the path to JAVA_HOME in the CMakeLists file, so back it up and do that
cp CMakeLists.txt CMakeLists.txt.bak
sed -i 's|__JAVA_HOME__|'$JAVA_HOME'|g' CMakeLists.txt || fail "Sed"

#Generate the Makefiles and Make it!
cmake . >/dev/null || fail "CMake"
make >/dev/null || fail "Make"
printf "${GREEN}OK${NC}\n"

#Create the JAR to include into other projects
printf "Compiling Java bindings..."
cd ..
./gradlew build >/dev/null || fail "Gradle"

printf "${GREEN}OK${NC}\n"

#Lastly we want to copy all the important bits together into one folder to make life easier for the user
printf "Finishing up..."
mkdir -p build
cp libdeepspeech/libdeepspeech-jni.so build/
cp libdeepspeech/libs/libdeepspeech.so build/
cp libdeepspeech/build/libs/libdeepspeech.jar build/

#Restore the original CMake file
mv libdeepspeech/CMakeLists.txt.bak libdeepspeech/CMakeLists.txt

printf "${GREEN}OK${NC}\n\n"

pwd=$(pwd)
echo "Done. Your files are located in ${pwd}/build/"
