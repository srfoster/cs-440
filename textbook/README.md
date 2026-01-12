# CS440: Artificial Intelligence Textbook

An interactive textbook for CS440: Artificial Intelligence, built with React and the textbook-lib framework.

## Getting Started

1. Install dependencies:
```bash
npm install
```

2. Compile the textbook content:
```bash
npm run compile
```

3. Run the development server:
```bash
npm run dev
```

4. Open your browser to the URL shown (typically http://localhost:5173)

## Project Structure

- `public/textbook/` - Textbook content (markdown and YAML files)
  - `index.md` - Main textbook page
  - `content/` - Module content and questions
- `src/` - React application source
  - `App.jsx` - Main application component
  - `compiled/` - Compiled textbook content (auto-generated)
- `scripts/` - Build scripts
  - `compile-content.js` - Compiles markdown and YAML into JavaScript

## Adding Content

1. Create markdown files in `public/textbook/content/`
2. Create question YAML files in module `questions/` folders
3. Update the `concept-map.yml` file to reference your questions
4. Run `npm run compile` to compile the content
5. Restart the dev server to see your changes

## Building for Production

```bash
npm run build
```

## Deploying to GitHub Pages

```bash
npm run deploy
```
