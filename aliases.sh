# Source this file to access the aliases

# Set constants and print them
ENV_DIR="~/capstone/envs"
echo "Using $ENV_DIR as environments directory"

# Environment Management aliases
# Update an existing environment
alias update_env="conda env update --file $ENV_DIR/environment.yaml"
# Export environment file for committing
alias export_env="conda env export --no-builds > $ENV_DIR/environment.yaml &&  $ENV_DIR/process_environment.sh"
# Export the complete environment information
alias export_full_env="conda env export > $ENV_DIR/full_environment.yaml"
