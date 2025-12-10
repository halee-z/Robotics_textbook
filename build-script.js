const { spawn, execSync } = require('child_process');
const fs = require('fs');
const path = require('path');

// Ensure the docusaurus binary has execution permissions
try {
  const docusaurusPath = './node_modules/.bin/docusaurus';
  if (fs.existsSync(docusaurusPath)) {
    fs.chmodSync(docusaurusPath, '755');
    console.log('Set execution permissions for docusaurus binary');
  } else {
    console.log('Docusaurus binary not found at default location');
  }
} catch (e) {
  console.log('Error setting permissions:', e.message);
}

// Try multiple approaches to execute the build
async function runBuild() {
  try {
    // First, try using the direct require approach
    console.log('Attempting to run Docusaurus build...');

    // Execute using npx which should handle the binary properly
    const buildProcess = spawn('npx', ['docusaurus', 'build'], {
      stdio: 'inherit',
      shell: process.platform === 'win32' ? false : true // Use shell on non-Windows platforms
    });

    buildProcess.on('close', (code) => {
      console.log(`Build process exited with code ${code}`);
      process.exit(code);
    });

    buildProcess.on('error', (error) => {
      console.error('Build process error:', error);
      process.exit(1);
    });
  } catch (error) {
    console.error('Error running build:', error);
    process.exit(1);
  }
}

runBuild();