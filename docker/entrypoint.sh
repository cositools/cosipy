#!/bin/bash -il

# Cosipy Container Entrypoint Script
# ----------------------------------
# This script is the entrypoint for the Cosipy Docker container. It sets up the 
# environment, creates the necessary user accounts, configures the shell for the 
# container's runtime, and ensures proper permissions and user settings. 
# The goal is to provide a clean, usable environment for Cosipy.

# Create a marker file in the user's home directory
MARKER_FILE="$HOME/.cosipy-container"

if [ ! -e "$MARKER_FILE" ]; then
  touch -d "1 minute ago" "$MARKER_FILE"
fi

# Create the wheel group if it doesn't exist
if ! getent group wheel >/dev/null; then
  groupadd wheel
fi

# Create the cosi user with the specified HOST_USER_ID or default to 9001
USER_ID=${HOST_USER_ID:-9001}
if ! id -u cosi &>/dev/null; then
  useradd --shell /bin/bash -u $USER_ID -o -c "" -m cosi
  usermod -aG sudo cosi
  usermod -aG wheel cosi
fi

# Set environment variables for the user
export HOME=/home/cosi
export USER=cosi
export LOGNAME=cosi
export COSIPY_DIR=/opt/cosipy/repo
export PATH=/opt/conda/bin:$PATH

# Ensure the COSIPy environment is activated on login
if [ ! -e "$HOME/.bash_profile" ] || [ "$HOME/.bash_profile" -ot "$MARKER_FILE" ]; then
  cat <<EOF >> "$HOME/.bash_profile"
#################
# Added by entrypoint at date: $(date)
# Environment variables
export HOME=${HOME}
export USER=${USER}
export LOGNAME=${LOGNAME}
export COSIPY_DIR=${COSIPY_DIR}
export PATH=${PATH}:\$PATH
EOF
fi

# Aliases and utilities
if [ ! -e "$HOME/.bashrc" ] || [ "$HOME/.bashrc" -ot "$MARKER_FILE" ]; then
  cat <<EOF >> "$HOME/.bashrc"
# Aliases
alias notebook='jupyter notebook --ip 0.0.0.0 --no-browser'

# Shell options and history settings
export HISTCONTROL=ignoreboth:erasedups
HISTSIZE=1000
HISTFILESIZE=2000

# SHOPT settings
shopt -s autocd   # change to named directory
shopt -s cdspell  # autocorrects cd misspellings
shopt -s cmdhist  # save multi-line commands in history as a single line
shopt -s dotglob
shopt -s histappend  # do not overwrite history
shopt -s expand_aliases
shopt -s checkwinsize # checks term size when bash regains control

# Case-insensitive tab completion
bind "set completion-ignore-case on"

# History completion using arrow keys
bind '"\e[A": history-search-backward'
bind '"\e[B": history-search-forward'
EOF
fi

echo "Welcome to Cosipy!"
echo "For instructions on how to use this container, follow the README in the repository."

cat /dev/null > ~/.bash_history && history -c

exec /bin/bash