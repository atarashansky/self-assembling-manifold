#!/bin/bash
if [ $TRAVIS_OS_NAME == 'linux' ]; then
  echo "Installing deps for linux"
  sudo apt-get update
  sudo apt-get install libhdf5-dev
  #sudo apt-get install libgcc-5-dev

  #sudo add-apt-repository ppa:nschloe/swig-backports -y
  #sudo apt-get -qq update
  #sudo apt-get install -y swig3.0
elif [ $TRAVIS_OS_NAME == 'osx' ]; then
  echo "Installing deps for OSX"
  if [ $PYTHON_VERSION == "2.7" ]; then
    CONDA_VER='2'
  elif [ $PYTHON_VERSION == "3.7" ]; then
    CONDA_VER='3'
  else
    echo "Miniconda only supports 2.7 and 3.7"
  fi
  curl "https://repo.continuum.io/miniconda/Miniconda${CONDA_VER}-latest-MacOSX-x86_64.sh" -o "miniconda.sh"
  bash "miniconda.sh" -b -p $HOME/miniconda
  echo "$PATH"
  export PATH="$HOME/miniconda/bin:$PATH"
  source $HOME/miniconda/bin/activate
  # Use pip from conda
  #conda install -y python.app
  conda install -y pip
  conda install -y matplotlib
  conda install -y h5py
  pip --version

else
  echo "OS not recognized: $TRAVIS_OS_NAME"
  exit 1
fi
