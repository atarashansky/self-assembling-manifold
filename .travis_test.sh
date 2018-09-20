#!/bin/bash
if [ "$TRAVIS_OS_NAME" == 'osx' ]; then
  export PATH="$HOME/miniconda/bin:$PATH"
  source $HOME/miniconda/bin/activate
  PYTHON="$HOME/miniconda/bin/python$PYTHON_VERSION"
else
  PYTHON=${PYTHON:-python}
fi

echo "python: ${PYTHON}"

echo 'Running tests...'

#echo 'Dataset tests...'
#${PYTHON} "test/test_dataset.py"
#if [ $? != 0 ]; then
#    exit 1
#fi
#FIXME
echo 'No tests for now, skipping...'

echo 'done!'
