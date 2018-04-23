#!/usr/bin/env bash

ARGS_COUNT=2

vars_count=$#
if (( $vars_count < $ARGS_COUNT )); then
  echo "Error: This shell script needs at least $ARGS_COUNT argument(s)!"
  exit 1
fi

flag=$1
filepath=$2
echo ${flag} > "${filepath}"

