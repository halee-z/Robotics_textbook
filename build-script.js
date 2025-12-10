const { spawn } = require('child_process');
const fs = require('fs');

// Ensure the docusaurus binary has execution permissions
try {
  fs.chmodSync('./node_modules/.bin/docusaurus', '755');
} catch (e) {
  console.log('chmod not needed or failed');
}

// Execute the build command
const buildProcess = spawn('npx', ['docusaurus', 'build'], {
  stdio: 'inherit',
  shell: true
});

buildProcess.on('close', (code) => {
  process.exit(code);
});