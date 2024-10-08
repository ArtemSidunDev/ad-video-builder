const sharp = require('sharp');
const fs = require('fs/promises');
const { exec } = require('child_process');
const { launch } = require('puppeteer');
const UserAgent = require('user-agents');

const width = 430;
const height = 932;
const deviceScaleFactor = 1;

const cleanup = async (screenshots) => {
    for (const file of screenshots) {
        const resultFs = await fs.access(file).catch(() => null);
        if(resultFs) {
            await fs.unlink(file);
        }
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

const startSiteProcessing = async (siteUrl, folderPath) => {
    
    const browser = await launch({args: ['--no-sandbox']});
    
    try {
        const page = await browser.newPage();
        
        const userAgent = new UserAgent();
        
        await page.emulate({
            viewport: {
                width,
                height,
                deviceScaleFactor,
                isMobile:    true,
                hasTouch:    true,
                isLandscape: false,
            },
            userAgent: userAgent.toString(),
        }); 
        await page.goto(siteUrl, {timeout: 1000 * 60 * 5, waitUntil: 'domcontentloaded'});

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
        
        return screenshots

    } catch (error) {
        console.error(error);
        browser.close();
        return [];
    }
};

const run = async (siteUrl, siteScrollResultVideoPath, folderPath, duration) => {
    console.log('Running site scroll');
    const screenshots = await startSiteProcessing(siteUrl, folderPath);
    console.log('Site scroll screenshots created');
    if(screenshots.length === 0) {
        const image1 = `${folderPath}/1.png`;
        const image2 = `${folderPath}/2.png`;
        const image3 = `${folderPath}/3.png`;

        const metadata1 = await sharp(image1).metadata()
        const metadata2 = await sharp(image2).metadata()
        const metadata3 = await sharp(image3).metadata()
        
        await sharp({
            create: {
              width: metadata1.width,
              height: metadata1.height + metadata2.height + metadata3.height,
              channels: 4,
              background: { r: 255, g: 255, b: 255, alpha: 0 },
            },
          })
          .composite([
            { input: image1, top: 0, left: 0 },
            { input: image2, top: metadata1.height, left: 0 },
            { input: image3, top: metadata1.height + metadata2.height, left: 0 },
          ])
          .toFile(`${folderPath}/screenshots1.png`)

        screenshots.push(`${folderPath}/screenshots1.png`);
        screenshots.push(`${folderPath}/screenshots1.png`);
        screenshots.push(`${folderPath}/screenshots1.png`);
    }
    await createVideo(screenshots, siteScrollResultVideoPath, duration);
    console.log('Site scroll video created');
    await cleanup(screenshots);
    console.log('Site scroll finished');
};

module.exports = {
    run
}