const originalCwd = process.cwd();
const path = require('path');
process.chdir(path.join(__dirname, '/'));
console.log('Changed working directory to project root');

// Set the permissions as early as possible in the Vercel build environment
const fs = require('fs');
const child_process = require('child_process');

// Try to set permissions on the docusaurus binary
try {
  const docusaurusPath = './node_modules/.bin/docusaurus';
  if (fs.existsSync(docusaurusPath)) {
    fs.chmodSync(docusaurusPath, 0o755);
    console.log('✓ Set execution permissions on docusaurus binary');
  }
} catch (e) {
  console.log('Could not set permissions on docusaurus binary:', e.message);
}

// Then try building in multiple ways
async function buildSite() {
  try {
    // Method 1: Try using require to directly call Docusaurus build function
    console.log('Attempting to build using direct require...');
    const docusaurus = require('@docusaurus/core');
    
    // If the direct require worked, call the build function
    if (docusaurus.build) {
      await docusaurus.build(process.cwd(), {});
      console.log('✓ Site built successfully with direct require');
      process.exit(0);
    } else {
      console.log('Docusaurus build function not available via direct require');
    }
  } catch (requireErr) {
    console.log('Failed to build with direct require:', requireErr.message);
  }

  try {
    // Method 2: Use the Node.js approach by directly calling the Docusaurus CLI
    console.log('Attempting to build using direct Node.js execution...');
    const cliPath = './node_modules/@docusaurus/core/bin/docusaurus.js';
    
    if (fs.existsSync(cliPath)) {
      const {spawn} = child_process;
      const buildProcess = spawn('node', [cliPath, 'build'], {
        stdio: 'inherit',
        env: {...process.env}
      });
      
      buildProcess.on('close', (code) => {
        if (code === 0) {
          console.log('✓ Site built successfully with direct Node.js execution');
          process.exit(0);
        } else {
          console.log(`Build failed with exit code: ${code}`);
          process.exit(code);
        }
      });
    } else {
      console.log('Docusaurus CLI not found at expected path:', cliPath);
    }
  } catch (nodeErr) {
    console.log('Failed to build with direct Node.js execution:', nodeErr.message);
  }

  try {
    // Method 3: Try npx with a different approach
    console.log('Attempting to build using npx...');
    const {spawn} = child_process;
    const buildProcess = spawn('npx', ['--yes', 'docusaurus', 'build'], {
      stdio: 'inherit',
      env: {...process.env}
    });
    
    buildProcess.on('close', (code) => {
      if (code === 0) {
        console.log('✓ Site built successfully with npx');
        process.exit(0);
      } else {
        console.log(`npx build failed with exit code: ${code}`);
        process.exit(code);
      }
    });
  } catch (npxErr) {
    console.log('Failed to build with npx:', npxErr.message);
  }
  
  // If all methods failed, try an alternative approach
  try {
    console.log('Trying last resort: inline build via Node API...');
    const build = require('@docusaurus/core/lib/commands/build').build;
    const config = require('./docusaurus.config');
    
    // Create a context object similar to what Docusaurus expects
    const context = {
      siteDir: process.cwd(),
      generatedFilesDir: path.join(process.cwd(), '.docusaurus'),
      siteConfig: config,
      outDir: path.join(process.cwd(), 'build'),
      baseUrl: config.baseUrl,
      i18n: {
        currentLocale: config.i18n.defaultLocale,
        locales: config.i18n.locales,
        isRTL: false
      }
    };
    
    await build(context);
    console.log('✓ Site built successfully with Node API');
    process.exit(0);
  } catch (apiErr) {
    console.log('Last resort also failed:', apiErr.message);
    process.exit(1);
  }
}

// Execute the build
buildSite();