#!/bin/bash

git add .
read -p "Comments: " comment
git commit -m "$comment"
git push