#!/bin/sh
# vercel-build.sh
# Ensure the docusaurus binary is executable if present, then run the build via npx.
# This is robust across different package managers / installer behaviors.
chmod +x ./node_modules/.bin/docusaurus 2>/dev/null || true

# Use npx to invoke Docusaurus (works even if the .bin file isn't executable)
npx docusaurus build
