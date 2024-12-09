const sharp = require('sharp');
const fs = require('fs/promises');
const { exec } = require('child_process');
const { launch } = require('puppeteer');
const UserAgent = require('user-agents');
const mp3Duration = require('mp3-duration');

const width = 430;
const height = 932;
const deviceScaleFactor = 1;

const popupConfig = [
    {
        popupSelector: '.pos-popup_center',
        closeButtonSelector: '.proof-factor-cb-prompt-close',
        removePopup: true
    }
];

const closePopups = async (page, popups) => {
    for (const popup of popups) {
        const { popupSelector, closeButtonSelector, removePopup } = popup;

        try {
            await page.waitForSelector(popupSelector, { timeout: 5000 });

            if (closeButtonSelector) {
                const closeButton = await page.$(closeButtonSelector);
                if (closeButton) {
                    await closeButton.click();
                    continue;
                } else {
                    try {
                        await page.evaluate((selector) => {
                            const popup = document.querySelector(selector);
                            if (popup) {
                                popup.style.display = 'none';
                                console.log(`Hide ${selector}`);
                            }
                        }, popupSelector);
                    } catch (error) {
                        console.log(`popupSelector, ${error.message}`);
                    }
                }
            }

            if (removePopup) {
                await page.evaluate((selector) => {
                    const popup = document.querySelector(selector);
                    if (popup) {
                        popup.style.display = 'none';
                        console.log(`Попап скрыт: ${selector}`);
                    }
                }, popupSelector);
            }
        } catch (error) {
            console.log('popupSelector', error.message);
        }
    }
};

const cleanup = async (screenshots) => {
    for (const file of screenshots) {
        const resultFs = await fs.access(file).catch(() => null);
        if(resultFs) {
            await fs.unlink(file);
        }
    }
};

const createVideo = async (screenshots, siteScrollResultVideoPath, duration, width, height) => {
    const images = screenshots.map(screenshot => sharp(screenshot));

    const metaData = await images[0].metadata();
    const totalHeight = metaData.height;
    const scrollSpeed = (totalHeight - 3840) / duration;

    const ffmpegCommand = `ffmpeg -loop 1 -i ${screenshots[0]} -filter_complex "[0:v]scale=${width}:-1,format=yuv420p,fps=120[v];color=size=${width}x${height}:rate=120:color=black[d];[d][v]overlay=shortest=1:y='-(t*${scrollSpeed})'" -t ${duration} -pix_fmt yuv420p -preset ultrafast ${siteScrollResultVideoPath}`;

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

async function getMp3Duration(filePath) {
    try {
        const duration = await new Promise((resolve, reject) => {
            mp3Duration(filePath, (err, duration) => {
                if (err) reject(err);
                else resolve(duration);
            });
        });
        return duration;
    } catch (error) {
        console.error('Помилка:', error.message);
    }
}

const startSiteProcessing = async (siteUrl, folderPath) => {
    let browser;
    try {
        const siteUrlLower = siteUrl.toLowerCase();
        if(siteUrlLower.includes('etsy') || siteUrlLower.includes('amazon') || siteUrlLower.includes('amzn')) {
            return [];
        }
        browser = await launch({args: ['--no-sandbox']});
        
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

        await closePopups(page, popupConfig);

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
        return [];
    }
};

const run = async (siteUrl, siteScrollResultVideoPath, folderPath, duration, voice=false) => {
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
          .toFile(`${folderPath}/screenshots2.png`)

        screenshots.push(`${folderPath}/screenshots2.png`);
    }
    await createVideo([`${folderPath}/screenshot2.png`], siteScrollResultVideoPath, duration, 2160, 3840);
    
    if(voice) {
        const image1 = `${folderPath}/1.png`;
        const image2 = `${folderPath}/2.png`;
        const image3 = `${folderPath}/3.png`;
        const image4 = `${folderPath}/4.png`;
        const image5 = `${folderPath}/5.png`;
        const image6 = `${folderPath}/6.png`;
        const image7 = `${folderPath}/7.png`;

        const metadata1 = await sharp(image1).metadata()
        const metadata2 = await sharp(image2).metadata()
        const metadata3 = await sharp(image3).metadata()
        const metadata4 = await sharp(image4).metadata()
        const metadata5 = await sharp(image5).metadata()
        const metadata6 = await sharp(image6).metadata()
        const metadata7 = await sharp(image7).metadata()
        
        await sharp({
            create: {
              width: metadata1.width,
              height: metadata1.height + metadata2.height + metadata3.height + metadata4.height + metadata5.height + metadata6.height + metadata7.height,
              channels: 4,
              background: { r: 255, g: 255, b: 255, alpha: 0 },
            },
          })
          .composite([
            { input: image1, top: 0, left: 0 },
            { input: image2, top: metadata1.height, left: 0 },
            { input: image3, top: metadata1.height + metadata2.height, left: 0 },
            { input: image4, top: metadata1.height + metadata2.height + metadata3.height, left: 0 },
            { input: image5, top: metadata1.height + metadata2.height + metadata3.height + metadata4.height, left: 0 },
            { input: image6, top: metadata1.height + metadata2.height + metadata3.height + metadata4.height + metadata5.height, left: 0 },
            { input: image7, top: metadata1.height + metadata2.height + metadata3.height + metadata4.height + metadata5.height + metadata6.height, left: 0 },
          ])
          .toFile(`${folderPath}/screenshots5.png`)

        screenshots.push(`${folderPath}/screenshots5.png`);
        
        const voiceDuration = await getMp3Duration(`${folderPath}/voice.mp3`);

        await createVideo([`${folderPath}/screenshots5.png`], `${folderPath}/bg_avatar.mp4`, voiceDuration, 1216, 2160);
    }

    console.log('Site scroll video created');
    await cleanup(screenshots);
    console.log('Site scroll finished');
};

module.exports = {
    run
}