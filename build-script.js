const { spawn, execSync } = require('child_process');
const fs = require('fs');
const path = require('path');

// Ensure the docusaurus binary has execution permissions
function setBinaryPermissions() {
  try {
    const docusaurusPath = './node_modules/.bin/docusaurus';

    if (fs.existsSync(docusaurusPath)) {
      // Try to set permissions using chmod command (works in Linux/Unix systems)
      try {
        execSync(`chmod +x "${path.resolve(docusaurusPath)}"`);
        console.log('Set execution permissions for docusaurus binary using chmod');
      } catch (chmodError) {
        console.log('Chmod failed, trying fs.chmod:', chmodError.message);
        // Fallback to fs.chmod
        fs.chmodSync(docusaurusPath, 0o755);
        console.log('Set execution permissions using fs.chmod');
      }
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
    env: { ...process.env,  // Pass through environment variables
           SKIP_PREFLIGHT_CHECK: 'true' } // Skip preflight checks that might cause issues
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