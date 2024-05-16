const { exec } = require('child_process');
const axios = require('axios');
const fs = require('fs');
const AWS = require('aws-sdk');

const { AWS_KEY_ID, AWS_SECRET_KEY, AWS_S3_AD_VIDEOS_BUCKET } = process.env;

AWS.config.update({
  accessKeyId: AWS_KEY_ID,
  secretAccessKey: AWS_SECRET_KEY
});

const s3 = new AWS.S3();

async function handle(templateName, data) {
  const {
    adVideoId,
    userId,
    media,
    avatarUrl,
    musicUrl
  } = data;
  const folderPath = `./data/${adVideoId}`;
  try {
    if (fs.existsSync(folderPath)) {
      fs.rm(folderPath, { recursive: true })
    }
    fs.mkdirSync(folderPath, { recursive: true });
    
    const siteScroll = media.find(mediaItem => mediaItem.url.includes('siteScrollResultVideo'));
    const videos = media.filter(mediaItem => mediaItem.type === 'video').filter(mediaItem => !mediaItem.url.includes('siteScrollResultVideo'));
    const images = media.filter(mediaItem => mediaItem.type === 'image');
    
    await Promise.all([
      download(siteScroll.url, `${folderPath}/ss.mp4`),
      download(avatarUrl, `${folderPath}/avatar.mp4`),
      download('https://user-advideos.s3.amazonaws.com/static/action.mp4', `${folderPath}/action.mp4`),
      download(musicUrl, `${folderPath}/background_audio.mp3`)
    ]);

    await generateSubtitles(folderPath);
    
    await Promise.all(videos.map(async (mediaItem, index) => {
      const mediaPath = `${folderPath}/${index+1}.mp4`;
      await download(mediaItem.url, mediaPath);
    }));

    await Promise.all(images.map(async (mediaItem, index) => {
      const mediaPath = `${folderPath}/${index+1}.png`;
      await download(mediaItem.url, mediaPath);
    }));

    await runCommand(`./templates/${templateName}/run.sh ${folderPath}`);
    
    console.log('start uploading to s3');

    const url = await uploadToS3(`${folderPath}/output.mp4`, `${userId}/${adVideoId}/${adVideoId}.mp4`);
    
    console.log(url);
    
    fs.rm(folderPath, { recursive: true }, (err) => {
      if (err) {
        console.error(err);
        return;
      }
    });
    
    return url
  } 
  catch (error) {
    console.log(error);
    fs.rm(folderPath, { recursive: true }, (err) => {
      if (err) {
        console.error(err);
        return;
      }
    });
    return false;
  }
}

function uploadToS3(mediaPath, s3Key) {
  const fileContent = fs.readFileSync(mediaPath);

  const params = {
    Bucket: AWS_S3_AD_VIDEOS_BUCKET,
    Key: s3Key,
    Body: fileContent,
    ContentType: 'video/mp4',
  };
  return new Promise((resolve, reject) => {
    s3.upload(params, function(err, data) {
      if (err) {
        reject(err);
      }
      console.log(`File uploaded successfully. ${data.Location}`);
      resolve(data.Location);
    });
  });
}

// download media function
async function download(mediaUrl, mediaPath) {
  const writer = fs.createWriteStream(mediaPath);
  const response = await axios({
    url: mediaUrl,
    method: 'GET',
    responseType: 'stream'
  });
  response.data.pipe(writer);
}

async function generateSubtitles(folderPath) {
  await runCommand(`whisper_timestamped ${folderPath}/avatar.mp4 --model tiny --output_dir ${folderPath}/words`);
  const data = fs.readFileSync(`${folderPath}/words/avatar.mp4.words.json`, 'utf8');
  const parsedData = JSON.parse(data);
  const words = [];
  for(const segment of parsedData.segments) {
    for(const word of segment.words) {
      words.push({
        start: word.start,
        end:   word.end,
        word:  word.text
      });
    }
  }

  fs.writeFileSync(`${folderPath}/transcription.json`, JSON.stringify(words), { encoding: 'utf8' });
  return `${folderPath}/transcription.json`;
}

async function runCommand(command) {
  console.log('\n');
  console.log(command);
  console.log('\n');
  return new Promise((resolve, reject) => {
    const test = exec(command, (error, stdout, stderr) => {
      if (error) {
        console.error(`exec error: ${error}`);
        console.error(`exec stderr: ${stderr}`);
        reject(error);
        return;
      }
      resolve(stdout);
    });

    test.stdout.on('data', (data) => {
      console.log('stdout: ' + data);
    });

    test.stderr.on('data', (data) => {
      console.log('stderr: ' + data);
    });
  });
}



module.exports = {
  handle
}