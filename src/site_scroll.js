const sharp = require('sharp');
const fs = require('fs/promises');
const { exec } = require('child_process');
const { launch } = require('puppeteer');

const width = 430;
const height = 932;
const deviceScaleFactor = 1;

const cleanup = async (screenshots) => {
    for (const file of screenshots) {
        await fs.unlink(file);
    }
};

const createVideo = async (screenshots, siteScrollResultVideoPath, duration) => {
    const images = screenshots.map(screenshot => sharp(screenshot));
    const metaData = await images[2].metadata();
    const totalHeight = metaData.height;
    const scrollSpeed = (totalHeight - 3840) / duration;

    const ffmpegCommand = `ffmpeg -loop 1 -i ${screenshots[2]} -filter_complex "[0:v]scale=2160:-1,format=yuv420p,fps=120[v];color=size=2160x3840:rate=120:color=black[d];[d][v]overlay=shortest=1:y='-(t*${scrollSpeed})'" -t ${duration} -pix_fmt yuv420p -preset ultrafast ${siteScrollResultVideoPath}`;

    return new Promise((resolve, reject) => {
        const data = exec(ffmpegCommand, (error, stdout, stderr) => {
            if (error) {
                console.error(error);
                console.error(stderr);
                console.error(stdout);
                reject(error);
                return;
            }
            resolve(siteScrollResultVideoPath);
        });
    });
};

const delay = (time) => {
    return new Promise(resolve => setTimeout(resolve, time));
};

const autoScroll = async (page) => {
    const scrollStep = 800;
    const scrollDelay = 500;
    const maxScrollAttempts = 500;

    let attempts = 0;
    let lastScrollPosition = 0;
    let currentScrollPosition = 0;

    while (true) {
        lastScrollPosition = await page.evaluate(() => window.scrollY);
        await page.evaluate((step) => window.scrollBy(0, step), scrollStep);
        await delay(scrollDelay);

        currentScrollPosition = await page.evaluate(() => window.scrollY);

        if (currentScrollPosition <= lastScrollPosition || attempts >= maxScrollAttempts) {
            break;
        }

        attempts++;
    }
    await page.evaluate(() => window.scrollTo(0, 0));
    await delay(1000);
    return attempts * scrollStep;
};

const run = async (siteUrl, siteScrollResultVideoPath, folderPath, duration) => {
    console.log('Running site scroll');
    const browser = await launch({args: ['--no-sandbox']});
    const page = await browser.newPage();

    await page.emulate({
        viewport: {
            width,
            height,
            deviceScaleFactor,
            isMobile:    true,
            hasTouch:    true,
            isLandscape: false,
        },
        userAgent: 'Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.150 Mobile Safari/537.36'
    });

    await page.goto(siteUrl, {timeout: 0});
    
    const screenshots = [];
    
    await delay(1000);
    
    await page.evaluate(() => {
        const element = document.querySelector('a[href="https://zeely.app?utm_source=link"]');
        if (element) {
            element.remove();
        }
    });

    const siteHeight = await autoScroll(page);
    await delay(1000);
    let i = 0;
    console.log('Taking screenshots');
    
    if(siteHeight > 10000) {
        page.setViewport({
            width,
            height: height * 7,
            deviceScaleFactor,
        });
        await page.screenshot({ path: `${folderPath}/screenshot${i}.png`, type: 'jpeg', quality: 100});
        screenshots.push(`${folderPath}/screenshot${i}.png`);
        i++;
        await delay(1000);
        await page.screenshot({ path: `${folderPath}/screenshot${i}.png`, type: 'jpeg', quality: 100});
        screenshots.push(`${folderPath}/screenshot${i}.png`);
        i++;
    } else {
        await page.screenshot({ path: `${folderPath}/screenshot${i}.png`, fullPage: true, type: 'jpeg', quality: 100 });
        screenshots.push(`${folderPath}/screenshot${i}.png`);
        i++;
        await delay(1000);
        await page.screenshot({ path: `${folderPath}/screenshot${i}.png`, fullPage: true, type: 'jpeg', quality: 100 });
        screenshots.push(`${folderPath}/screenshot${i}.png`);
        i++;
    }

    const dataImage = await sharp(`${folderPath}/screenshot${i-1}.png`).metadata();
    
    if(dataImage.height > 10000) {
        await sharp(`${folderPath}/screenshot${i-1}.png`)
        .resize(width, height * 7, {
            position: 'bottom'
        })
        .toFile(`${folderPath}/screenshot${i}.png`);
        i++;
    }

    await sharp(`${folderPath}/screenshot${i-1}.png`)
    .resize(2160)
    .png({
        quality:           100,
        compressionLevel:  0,
        adaptiveFiltering: true,
        effort:            6,
    })
    .toFile(`${folderPath}/screenshot${i}.png`);
    
    screenshots.push(`${folderPath}/screenshot${i}.png`);
    console.log('Screenshots taken');
    await browser.close();
    console.log('Site scroll screenshots created');
    await createVideo(screenshots, siteScrollResultVideoPath, duration);
    console.log('Site scroll video created');
    await cleanup(screenshots);
    console.log('Site scroll finished');
};

module.exports = {
    run
}