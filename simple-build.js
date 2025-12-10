const { spawn } = require('child_process');

// Execute the build using npx to ensure proper resolution in Vercel
function runBuild() {
  console.log('Attempting to run Docusaurus build using npx...');

  const buildProcess = spawn('npx', ['docusaurus', 'build'], {
    stdio: 'inherit',
    shell: true, // Using shell to help with permission issues
    env: { 
      ...process.env,
      SKIP_PREFLIGHT_CHECK: 'true'
    }
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