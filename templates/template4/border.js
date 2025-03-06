const sharp = require('sharp');
const fs = require('fs');
const path = require('path');

const inputFolder = process.argv[2];
if (!inputFolder) {
    console.error('❌ Укажите путь к папке с изображениями!');
    process.exit(1);
}

const borderSize = 20;
const bottomBorderSize = borderSize * 6;

async function addBorder(inputPath, outputPath) {
    try {
        const image = sharp(inputPath);
        const { width, height } = await image.metadata();

        const borderColor = { r: 255, g: 255, b: 255, alpha: 1 };

        await sharp({
            create: {
                width: width + 2 * borderSize,
                height: height + borderSize + bottomBorderSize,
                channels: 4,
                background: borderColor
            }
        })
        .composite([{ input: inputPath, top: borderSize, left: borderSize }])
        .toFile(outputPath);

        console.log(`✅ ${path.basename(inputPath)} (Рамка добавлена)`);
    } catch (err) {
        console.error(`❌ Error ${path.basename(inputPath)}:`, err);
    }
}

// Обрабатываем файлы
async function processImages() {
    await addBorder(`${inputFolder}/1.png`, `${inputFolder}/1-bordered.png`);
    await addBorder(`${inputFolder}/2.png`, `${inputFolder}/2-bordered.png`);
    await addBorder(`${inputFolder}/3.png`, `${inputFolder}/3-bordered.png`);
}

processImages();