#!/bin/bash

# Basic CLI test script for peakachu_cohort

# Assumes the package is installed, or you have an entry point script.
# Replace 'peakachu-cohort' with the actual command if different.
# (e.g., 'python -m peakachu_cohort.cli' if running from source)
CLI_COMMAND="peakachu-cohort"

echo "Running basic CLI tests..."

# Test --help flag
echo "--- Testing --help ---"
$CLI_COMMAND --help
if [ $? -ne 0 ]; then
    echo "Error: --help flag failed!"
    exit 1
fi
echo "--help OK"

# Test --version flag
echo "--- Testing --version ---"
$CLI_COMMAND --version
if [ $? -ne 0 ]; then
    echo "Error: --version flag failed!"
    exit 1
fi
echo "--version OK"

# Test subcommand help
echo "--- Testing process --help ---"
$CLI_COMMAND process --help
if [ $? -ne 0 ]; then
    echo "Error: process --help failed!"
    exit 1
fi
echo "process --help OK"

echo "--- Testing analyze --help ---"
$CLI_COMMAND analyze --help
if [ $? -ne 0 ]; then
    echo "Error: analyze --help failed!"
    exit 1
fi
echo "analyze --help OK"

echo "--- Testing visualize --help ---"
$CLI_COMMAND visualize --help
if [ $? -ne 0 ]; then
    echo "Error: visualize --help failed!"
    exit 1
fi
echo "visualize --help OK"

echo "--- Testing validate --help ---"
$CLI_COMMAND validate --help
if [ $? -ne 0 ]; then
    echo "Error: validate --help failed!"
    exit 1
fi
echo "validate --help OK"

# Placeholder test for running a command (requires setup)
# echo "--- Testing placeholder command (dry run) ---"
# $CLI_COMMAND process train --dry-run # Assuming a dry-run option exists
# if [ $? -ne 0 ]; then
#     echo "Error: Placeholder command test failed!"
#     exit 1
# fi
# echo "Placeholder command OK"

echo "Basic CLI tests passed!"
exit 0 