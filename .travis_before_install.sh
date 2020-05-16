#!/bin/bash
if [ $TRAVIS_OS_NAME == 'linux' ]; then
  echo "Installing deps for linux"
  sudo apt-get update
  sudo apt-get install libhdf5-dev
  #sudo apt-get install libgcc-5-dev

  #sudo add-apt-repository ppa:nschloe/swig-backports -y
  #sudo apt-get -qq update
  #sudo apt-get install -y swig3.0
else
  echo "OS not recognized: $TRAVIS_OS_NAME"
  exit 1
fi
