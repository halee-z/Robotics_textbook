const { spawn, execSync } = require('child_process');
const fs = require('fs');
const path = require('path');

// Function to check if running in Vercel environment
function isVercel() {
  return process.env.VERCEL === '1' || process.env.NOW_BUILDER === '1';
}

// Ensure the docusaurus binary has execution permissions
function setBinaryPermissions() {
  try {
    const docusaurusPath = path.join(__dirname, 'node_modules', '.bin', 'docusaurus');
    
    if (fs.existsSync(docusaurusPath)) {
      // Try to set permissions using chmod command
      try {
        execSync(`chmod +x "${docusaurusPath}"`);
        console.log('✓ Set execution permissions for docusaurus binary using chmod');
      } catch (chmodError) {
        console.log('Chmod failed, trying fs.chmod:', chmodError.message);
        try {
          // Fallback to fs.chmod
          fs.chmodSync(docusaurusPath, 0o755);
          console.log('✓ Set execution permissions using fs.chmod');
        } catch (fsError) {
          console.log('fs.chmod also failed:', fsError.message);
        }
      }
    } else {
      console.log('⚠ Docusaurus binary not found at default location');
    }
  } catch (e) {
    console.log('Error setting permissions:', e.message);
  }
}

// Execute the build using multiple fallback strategies
function runBuild() {
  console.log('Environment:', isVercel() ? 'Vercel' : 'Local');
  console.log('Setting binary permissions...');
  setBinaryPermissions();

  console.log('Attempting to run Docusaurus build...');
  
  // Try different strategies to execute the build command
  const buildProcess = spawn('npx', ['docusaurus', 'build'], {
    stdio: 'inherit',
    shell: true, // Using shell to help with permission issues
    env: {
      ...process.env,
      SKIP_PREFLIGHT_CHECK: 'true',
      // Set environment variables that might help with permissions
      npm_config_scripts_prepend_node_path: 'auto',
      PATH: `${path.join(__dirname, 'node_modules', '.bin')}:${process.env.PATH}`
    }
  });

  buildProcess.on('close', (code) => {
    console.log(`Build process exited with code ${code}`);
    // Only exit with error code if there was an actual error (not 0)
    if (code !== 0) {
      process.exit(code);
    }
  });

  buildProcess.on('error', (error) => {
    console.error('Build process error:', error);
    // Try fallback approach
    console.log('Trying fallback approach with direct node execution...');
    
    try {
      const docusaurusCliPath = path.join(__dirname, 'node_modules', '@docusaurus', 'core', 'bin', 'docusaurus.js');
      if (fs.existsSync(docusaurusCliPath)) {
        const fallbackProcess = spawn('node', [docusaurusCliPath, 'build'], {
          stdio: 'inherit',
          env: { ...process.env, SKIP_PREFLIGHT_CHECK: 'true' }
        });
        
        fallbackProcess.on('close', (fallbackCode) => {
          console.log(`Fallback build process exited with code ${fallbackCode}`);
          process.exit(fallbackCode);
        });
      } else {
        console.error('Fallback: Docusaurus CLI not found at expected location');
        process.exit(1);
      }
    } catch (fallbackError) {
      console.error('Fallback also failed:', fallbackError);
      process.exit(1);
    }
  });
}

// Run the build
runBuild();