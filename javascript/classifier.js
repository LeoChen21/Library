import { pipeline } from "@xenova/transformers";
import fetch from "node-fetch";
import * as cheerio from "cheerio";

// Function to extract text content from a webpage
async function extractTextFromWebsite(url) {
    try {
        const response = await fetch(url, { headers: { "User-Agent": "Mozilla/5.0" } });
        const html = await response.text();
        const $ = cheerio.load(html);

        // Extract text from paragraphs
        let pageText = $("p").map((_, el) => $(el).text()).get().join(" ");
        return pageText.substring(0, 1000); // Limit text length
    } catch (error) {
        console.error(`Failed to fetch ${url}: ${error.message}`);
        return null;
    }
}

// Function to classify website content
async function classifyWebsite(url) {
    const textContent = await extractTextFromWebsite(url);
    if (!textContent) {
        console.log(`${url}: No readable text found.`);
        return;
    }

    // Load the Hugging Face zero-shot classifier
    const classifier = await pipeline("zero-shot-classification");
    const mediaCategories = ["Anime", "Manga", "Novel"];

    // Classify the content
    const classification = await classifier(textContent, mediaCategories);
    const predictedLabel = classification.labels[0]; // Top prediction

    console.log(`${url}: ${predictedLabel}`);
}

// List of websites to classify
const websites = [
    "https://www.crunchyroll.com/",  // Likely Anime
    "https://www.viz.com/shonenjump", // Likely Manga
    "https://www.webnovel.com/",  // Likely Novel
];

// Process and classify websites
(async () => {
    for (const site of websites) {
        await classifyWebsite(site);
    }
})();
