#!/bin/bash
set -e

# Ensure proper permissions for docusaurus binary
chmod +x node_modules/.bin/docusaurus || echo "Permission setting not needed or failed"

# Run the build command
npx docusaurus build