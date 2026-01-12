import fs from 'fs';
import path from 'path';
import yaml from 'js-yaml';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Configuration
const TEXTBOOK_DIR = path.resolve(__dirname, '../public/textbook');
const OUTPUT_DIR = path.resolve(__dirname, '../src/compiled');

console.log('🚀 Starting content compilation...');
console.log('📁 Textbook directory:', TEXTBOOK_DIR);
console.log('📁 Output directory:', OUTPUT_DIR);

// Ensure directories exist
if (!fs.existsSync(TEXTBOOK_DIR)) {
  console.error('❌ Textbook directory not found:', TEXTBOOK_DIR);
  process.exit(1);
}

// Clean and create output directory
console.log('🧹 Cleaning output directory...');
if (fs.existsSync(OUTPUT_DIR)) {
  fs.rmSync(OUTPUT_DIR, { recursive: true, force: true });
}
fs.mkdirSync(OUTPUT_DIR, { recursive: true });
console.log('✅ Cleaned output directory');

// Storage for compiled content
const compiledFiles = {};
let yamlCount = 0;
let markdownCount = 0;

// Helper to get all files recursively
function getAllFiles(dir, fileList = [], baseDir = dir) {
  const files = fs.readdirSync(dir);
  
  files.forEach(file => {
    const filePath = path.join(dir, file);
    const stat = fs.statSync(filePath);
    
    if (stat.isDirectory()) {
      getAllFiles(filePath, fileList, baseDir);
    } else {
      const relativePath = path.relative(baseDir, filePath).replace(/\\/g, '/');
      fileList.push(relativePath);
    }
  });
  
  return fileList;
}

// Compile YAML files
console.log('📄 Compiling YAML files...');
const yamlFiles = getAllFiles(TEXTBOOK_DIR).filter(f => f.endsWith('.yml'));
yamlFiles.forEach(relativePath => {
  console.log(`  Processing: ${relativePath}`);
  const fullPath = path.join(TEXTBOOK_DIR, relativePath);
  const content = fs.readFileSync(fullPath, 'utf8');
  const parsed = yaml.load(content);
  
  compiledFiles[relativePath] = {
    type: 'yaml',
    module: parsed
  };
  yamlCount++;
});
console.log(`✅ Compiled ${yamlCount} YAML files`);

// Compile Markdown files
console.log('📝 Compiling Markdown files...');
const markdownFiles = getAllFiles(TEXTBOOK_DIR).filter(f => f.endsWith('.md'));
markdownFiles.forEach(relativePath => {
  console.log(`  Processing: ${relativePath}`);
  const fullPath = path.join(TEXTBOOK_DIR, relativePath);
  const content = fs.readFileSync(fullPath, 'utf8');
  
  compiledFiles[relativePath] = {
    type: 'markdown',
    module: content
  };
  markdownCount++;
});
console.log(`✅ Compiled ${markdownCount} Markdown files`);

// Generate index.js with inlined content
console.log('📋 Generating index...');
const indexContent = `// Auto-generated compiled content - DO NOT EDIT
// Generated on ${new Date().toISOString()}

export const compiledFiles = ${JSON.stringify(compiledFiles, null, 2)};

export const stats = {
  yamlCount: ${yamlCount},
  markdownCount: ${markdownCount},
  totalCount: ${yamlCount + markdownCount}
};
`;

const outputPath = path.join(OUTPUT_DIR, 'index.js');
fs.writeFileSync(outputPath, indexContent);
console.log(`✅ Generated ${outputPath}`);

console.log('\n🎉 Compilation complete!');
console.log(`📊 Stats: ${yamlCount} YAML files, ${markdownCount} Markdown files`);
