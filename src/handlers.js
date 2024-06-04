const { exec } = require('child_process');
const axios = require('axios');
const fs = require('fs');
const AWS = require('aws-sdk');
const sharp = require('sharp');
const { run: runSiteScroll } = require('./site_scroll.js');

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
    callBackUrl,
    errorCallBackUrl,
    productUrl
  } = data;
  const folderPath = `./data/${adVideoId}`;
  if (fs.existsSync(folderPath)) {
    fs.rm(folderPath, { recursive: true })
  }
  fs.mkdirSync(folderPath, { recursive: true });
  try {
    await Promise.all([
      runSiteScroll(productUrl, `${folderPath}/ss.mp4`, folderPath, 25),
      prepare(folderPath, data)
    ]);

    await runCommand(`./templates/${templateName}/run.sh ${folderPath}`);
    const coverPath = await createCover(`${folderPath}/output.mp4`, folderPath);
    const url = await uploadToS3(`${folderPath}/output.mp4`, `${userId}/${adVideoId}/${adVideoId}.mp4`);
    const coverUrl = await uploadToS3(coverPath, `${userId}/${adVideoId}/${adVideoId}_cover.png`, 'image/png');

    fs.rm(folderPath, { recursive: true }, (err) => {
      if (err) {
        console.error(err);
        return;
      }
    });

    await axios.patch(callBackUrl, {
      url,
      coverUrl,
      status: 'done'
    })

    return true;
  } 
  catch (error) {
    console.error(error);
    fs.rm(folderPath, { recursive: true }, (err) => {
      if (err) {
        console.error(err);
        return;
      }
    });
    await axios.patch(errorCallBackUrl, {
      error,
      status: 'error'
    })
  }
}

async function prepare(folderPath, data) {
  const {
    media,
    avatarUrl,
    actionUrl,
    musicUrl
  } = data;
  
  const videos = media.filter(mediaItem => mediaItem.type === 'video');
  const images = media.filter(mediaItem => mediaItem.type === 'image');
  
  await Promise.all([
    download(avatarUrl, `${folderPath}/avatar.mp4`),
    download(actionUrl, `${folderPath}/action.mp4`),
    download(musicUrl, `${folderPath}/background_audio.mp3`),
    videos.map(async (mediaItem, index) => {
      const mediaPath = `${folderPath}/${index+1}.mp4`;
      await download(mediaItem.url, mediaPath);
    }),
    images.map(async (mediaItem, index) => {
      const mediaPathOrg = `${folderPath}/${index+1}_org.png`;
      const mediaPath = `${folderPath}/${index+1}.png`;
      await download(mediaItem.url, mediaPathOrg);
  
      await sharp(mediaPathOrg)
      .resize(2160)
      .jpeg({ quality: 100 })
      .toFile(mediaPath);
      fs.unlinkSync(mediaPathOrg);
    })
  ]);

  await runCommand(`ffmpeg -i ${folderPath}/avatar.mp4 -vf "setpts=0.7143*PTS" -filter:a "atempo=1.4" ${folderPath}/avatar_speed.mp4`);

  await runCommand(`ffmpeg -i ${folderPath}/avatar_speed.mp4 -filter:a "volume=2.0" ${folderPath}/avatar_speed_sound.mp4`);

  await runCommand(`ffmpeg -i ${folderPath}/avatar_speed_sound.mp4 -vf "scale=1080:1920" -b:v 2M ${folderPath}/avatar_end.mp4`);
  
  fs.unlinkSync(`${folderPath}/avatar_speed.mp4`);
  fs.unlinkSync(`${folderPath}/avatar_speed_sound.mp4`);
  fs.renameSync(`${folderPath}/avatar.mp4`, `${folderPath}/avatar_org.mp4`);
  fs.renameSync(`${folderPath}/avatar_end.mp4`, `${folderPath}/avatar.mp4`);

  await runCommand(`./templates/common/run.sh ${folderPath}`);

  await generateSubtitles(folderPath);
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
  fs.rm(`${folderPath}/words`, { recursive: true }, (err) => {
    if (err) {
      console.error(err);
      return;
    }
  });
  return `${folderPath}/transcription.json`;
}

async function createCover(videoPath, folderPath) {
  const outputPath = `${folderPath}/output_cover.png`;
  await runCommand(`ffmpeg -ss 1 -i ${videoPath} -vframes 1 ${outputPath}`);
  return outputPath;
}

function uploadToS3(mediaPath, s3Key, contentType = 'video/mp4') {
  const fileContent = fs.readFileSync(mediaPath);

  const params = {
    Bucket: AWS_S3_AD_VIDEOS_BUCKET,
    Key: s3Key,
    Body: fileContent,
    ContentType: contentType,
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

async function download(mediaUrl, mediaPath) {
  const writer = fs.createWriteStream(mediaPath);
  const response = await axios({
    url: mediaUrl,
    method: 'GET',
    responseType: 'stream'
  });
  response.data.pipe(writer);
  return new Promise((resolve, reject) => {
    writer.on('finish', resolve);
    writer.on('error', reject);
  });
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