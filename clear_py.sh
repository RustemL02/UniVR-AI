#!/bin/bash

# Removes all __pycache__ directories in this project.
find . | grep -E "(__pycache__|\.pyc$)" | xargs rm -rf
