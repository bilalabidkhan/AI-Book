# Quickstart: ROS 2 Humanoid Integration Module

## Prerequisites

- Node.js 18+ installed
- npm or yarn package manager
- Basic understanding of ROS 2 concepts (optional but helpful)

## Setup Instructions

### 1. Clone or Initialize the Repository

```bash
# If starting fresh
npx create-docusaurus@latest my-ai-book classic
cd my-ai-book
```

### 2. Install Dependencies

```bash
npm install
# or
yarn install
```

### 3. Create Module Directory Structure

```bash
mkdir -p docs/modules/1-ros2-humanoid-integration
```

### 4. Add Module Content Files

Create the three chapter files in the module directory:

```bash
touch docs/modules/1-ros2-humanoid-integration/ros2-fundamentals.md
touch docs/modules/1-ros2-humanoid-integration/python-agents-ros2.md
touch docs/modules/1-ros2-humanoid-integration/humanoid-modeling-urdf.md
```

### 5. Configure Docusaurus

Update `docusaurus.config.js` to include the new module:

```javascript
// docusaurus.config.js
module.exports = {
  // ... other config
  presets: [
    [
      'classic',
      {
        docs: {
          sidebarPath: require.resolve('./sidebars.js'),
          // ... other docs config
        },
        // ... other presets
      },
    ],
  ],
};
```

### 6. Update Sidebar Configuration

Update `sidebars.js` to include the new module:

```javascript
// sidebars.js
module.exports = {
  docs: [
    {
      type: 'category',
      label: 'Module 1: The Robotic Nervous System (ROS 2)',
      items: [
        'modules/1-ros2-humanoid-integration/ros2-fundamentals',
        'modules/1-ros2-humanoid-integration/python-agents-ros2',
        'modules/1-ros2-humanoid-integration/humanoid-modeling-urdf'
      ],
    },
    // ... other sidebar items
  ],
};
```

### 7. Run Development Server

```bash
npm run start
# or
yarn start
```

Your module will be available at http://localhost:3000.

## Content Creation Guidelines

### Writing for the Module

1. Each chapter should follow the format:
   - Introduction explaining the purpose
   - Main content with technical explanations
   - Practical examples where applicable
   - Summary of key concepts
   - Next steps or related topics

2. For code examples, use proper language annotation:
   ```markdown
   ```python
   # Python code example
   import rclpy
   from rclpy.node import Node
   ```
   ```

3. Maintain Flesch-Kincaid grade 10-12 readability level by:
   - Using active voice
   - Breaking complex concepts into smaller sections
   - Providing clear examples and analogies

### Verification Steps

1. Test all code examples in appropriate environments
2. Verify all links and cross-references work correctly
3. Ensure content meets technical accuracy standards
4. Confirm writing level compliance
5. Validate navigation and user experience

## Deployment

To build and deploy to GitHub Pages:

```bash
GIT_USER=<your-github-username> npm run deploy
```

Or for a general build:

```bash
npm run build
```

The built site will be in the `build/` directory and can be served statically.