const fs = require('fs');
const path = require('path');

/**
 * Script to update RAG (Retrieval-Augmented Generation) chatbot index
 * with new VLA (Vision-Language-Action) module content
 */

class RAGIndexUpdater {
    constructor(docsPath, indexPath) {
        this.docsPath = docsPath;
        this.indexPath = indexPath;
        this.vlaDocsPath = path.join(docsPath, 'module-4-vla');
    }

    async updateIndex() {
        console.log('Updating RAG chatbot index with VLA content...');

        // Read all VLA documentation files
        const vlaFiles = await this.getVLAFiles();

        // Extract content from each file
        const vlaContent = await this.extractContent(vlaFiles);

        // Update the search index
        await this.updateSearchIndex(vlaContent);

        // Update vector database for semantic search
        await this.updateVectorDatabase(vlaContent);

        console.log('RAG index updated successfully with VLA content');
    }

    async getVLAFiles() {
        const files = fs.readdirSync(this.vlaDocsPath);
        return files.filter(file => file.endsWith('.md')).map(file =>
            path.join(this.vlaDocsPath, file)
        );
    }

    async extractContent(filePaths) {
        const content = [];

        for (const filePath of filePaths) {
            const fileContent = fs.readFileSync(filePath, 'utf8');

            // Extract frontmatter and content
            const frontmatterMatch = fileContent.match(/^---\n([\s\S]*?)\n---/);
            const frontmatter = frontmatterMatch ? frontmatterMatch[1] : '';
            const body = fileContent.replace(/^---\n[\s\S]*?\n---\n/, '');

            // Parse frontmatter
            const metadata = this.parseFrontmatter(frontmatter);

            content.push({
                file: path.basename(filePath),
                title: metadata.title || 'Untitled',
                description: metadata.description || '',
                content: body,
                path: filePath,
                lastModified: fs.statSync(filePath).mtime
            });
        }

        return content;
    }

    parseFrontmatter(frontmatter) {
        if (!frontmatter) return {};

        const metadata = {};
        const lines = frontmatter.split('\n');

        for (const line of lines) {
            const colonIndex = line.indexOf(':');
            if (colonIndex > 0) {
                const key = line.substring(0, colonIndex).trim();
                const value = line.substring(colonIndex + 1).trim().replace(/^["']|["']$/g, '');
                metadata[key] = value;
            }
        }

        return metadata;
    }

    async updateSearchIndex(content) {
        // Simulate updating a search index (e.g., Elasticsearch, Algolia, etc.)
        console.log('Updating search index...');

        // In a real implementation, this would call the search API
        // For example: await elasticSearch.indexDocuments(content);

        // Create/update a simple JSON index file
        const indexData = {
            lastUpdated: new Date().toISOString(),
            documents: content.map(item => ({
                id: item.file,
                title: item.title,
                description: item.description,
                content: this.cleanContent(item.content),
                path: item.path,
                lastModified: item.lastModified
            }))
        };

        fs.writeFileSync(
            path.join(this.indexPath, 'vla-search-index.json'),
            JSON.stringify(indexData, null, 2)
        );

        console.log('Search index updated');
    }

    async updateVectorDatabase(content) {
        // Simulate updating a vector database for semantic search
        console.log('Updating vector database...');

        // In a real implementation, this would:
        // 1. Chunk the content into smaller pieces
        // 2. Generate embeddings using an ML model
        // 3. Store embeddings in a vector database (Pinecone, Weaviate, etc.)

        const vectorIndex = {
            lastUpdated: new Date().toISOString(),
            embeddings: content.map(item => ({
                id: item.file,
                title: item.title,
                chunks: this.chunkContent(this.cleanContent(item.content)),
                path: item.path
            }))
        };

        fs.writeFileSync(
            path.join(this.indexPath, 'vla-vector-index.json'),
            JSON.stringify(vectorIndex, null, 2)
        );

        console.log('Vector database updated');
    }

    chunkContent(content, chunkSize = 1000) {
        // Simple content chunking for semantic search
        const chunks = [];
        const paragraphs = content.split('\n\n');

        let currentChunk = '';
        for (const paragraph of paragraphs) {
            if (currentChunk.length + paragraph.length > chunkSize && currentChunk) {
                chunks.push(currentChunk.trim());
                currentChunk = paragraph;
            } else {
                currentChunk += '\n\n' + paragraph;
            }
        }

        if (currentChunk) {
            chunks.push(currentChunk.trim());
        }

        return chunks.filter(chunk => chunk.length > 50); // Filter out very small chunks
    }

    cleanContent(content) {
        // Remove markdown formatting for indexing
        return content
            .replace(/```[\s\S]*?```/g, '') // Remove code blocks
            .replace(/`[^`]*`/g, '') // Remove inline code
            .replace(/\[([^\]]+)\]\([^)]+\)/g, '$1') // Remove markdown links
            .replace(/#+\s/g, '') // Remove headers
            .replace(/\*\*([^*]+)\*\*/g, '$1') // Remove bold
            .replace(/\*([^*]+)\*/g, '$1') // Remove italic
            .replace(/^\s*[-*]\s/gm, '') // Remove list markers
            .replace(/\n{3,}/g, '\n\n') // Replace multiple newlines with double
            .trim();
    }
}

// Usage
async function main() {
    const updater = new RAGIndexUpdater(
        './docs',           // Documentation path
        './search-index'    // Index storage path
    );

    try {
        await updater.updateIndex();
        console.log('VLA content successfully indexed for RAG chatbot');
    } catch (error) {
        console.error('Error updating RAG index:', error);
        process.exit(1);
    }
}

if (require.main === module) {
    main();
}

module.exports = RAGIndexUpdater;