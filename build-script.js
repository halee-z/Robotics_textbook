const { spawn } = require('child_process');
const fs = require('fs');

// Ensure the docusaurus binary has execution permissions
function setBinaryPermissions() {
  try {
    const docusaurusPath = './node_modules/.bin/docusaurus';
    if (fs.existsSync(docusaurusPath)) {
      // Try to set permissions using chmod (works on Linux/Mac) or fs.chmod on all platforms
      fs.chmodSync(docusaurusPath, 0o755); // Use octal notation for permissions
      console.log('Set execution permissions for docusaurus binary');
    } else {
      console.log('Docusaurus binary not found at default location');
    }
  } catch (e) {
    console.log('Error setting permissions:', e.message);
  }
}

// Execute the build using npx to ensure proper resolution
function runBuild() {
  console.log('Setting binary permissions...');
  setBinaryPermissions();

  console.log('Attempting to run Docusaurus build...');

  // Execute using npx to ensure proper module resolution in Vercel environment
  const buildProcess = spawn('npx', ['docusaurus', 'build'], {
    stdio: 'inherit',
    env: { ...process.env } // Pass through environment variables
  });

  buildProcess.on('close', (code) => {
    console.log(`Build process exited with code ${code}`);
    process.exit(code);
  });

  buildProcess.on('error', (error) => {
    console.error('Build process error:', error);
    process.exit(1);
  });
}

// Run the build
runBuild();