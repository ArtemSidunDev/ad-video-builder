const { exec } = require('child_process');
const axios = require('axios');
const fs = require('fs');
const AWS = require('aws-sdk');
const sharp = require('sharp');
const { v4: uuidv4 } = require('uuid');
const { run: runSiteScroll } = require('./site_scroll.js');
const { run: generateSubtitlesSRT } = require('./subtitles_srt.js');
const { run: generateSubtitlesASS } = require('./subtitles_ass.js');

const { 
  AWS_KEY_ID, 
  AWS_SECRET_KEY, 
  AWS_S3_AD_VIDEOS_BUCKET, 
  AWS_S3_AD_VIDEOS_CLOUD_FRONT_URL 
} = process.env;

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
    productUrl,
    subtitleSettings = {},
    statusUpdateCallBackUrl
  } = data;
  console.log('Processing videos', adVideoId, userId)
  console.time('BUILD_TIME');
  //TODO: Add error handling
  const folderPath = `./data/${uuidv4()}`;
  if (fs.existsSync(folderPath)) {
    fs.rmSync(`${folderPath}`, { recursive: true }, (err) => {
      if (err) {
        console.error(err);
        return;
      }
    });
  }

  if(statusUpdateCallBackUrl) {
    await axios.patch(statusUpdateCallBackUrl, {
      status: 'BUILDING'
    })
  }

  fs.mkdirSync(folderPath, { recursive: true });
  
  const subtitleSettingsPath = `${folderPath}/subtitleSettings.json`
  
  fs.writeFileSync(subtitleSettingsPath, JSON.stringify(subtitleSettings, null, 2))

  try {
    
    await prepareMedia(folderPath, data);

    await Promise.all([
      runSiteScroll(productUrl, `${folderPath}/ss.mp4`, folderPath, 25),
      prepare(folderPath, data)
    ]);

    await runCommand(`./templates/${templateName}/run.sh ${folderPath} ${subtitleSettingsPath}`);
    
    const fileName = uuidv4();
    
    await runCommand(`ffmpeg -i ${folderPath}/output.mp4 -vf "scale=1152:2048" ${folderPath}/output_scale_command.mp4`);
    await runCommand(`ffmpeg -i ${folderPath}/output.mp4 -vf "scale=720:1280" ${folderPath}/output_low_scale_command.mp4`);
    
    fs.unlinkSync(`${folderPath}/output.mp4`);
    fs.renameSync(`${folderPath}/output_scale_command.mp4`, `${folderPath}/output.mp4`);
    
    const coverPath = await createCover(`${folderPath}/output.mp4`, folderPath);
    
    console.timeEnd('BUILD_TIME');

    const videoFilePath = `${userId}/${adVideoId}/${fileName}.mp4`
    const videoLowFilePath = `${userId}/${adVideoId}/${fileName}_low.mp4`
    const coverVideoFilePath = `${userId}/${adVideoId}/${fileName}_cover.png`
    
    await uploadToS3(`${folderPath}/output.mp4`, videoFilePath);
    await uploadToS3(`${folderPath}/output_low_scale_command.mp4`, videoLowFilePath);
    await uploadToS3(coverPath, coverVideoFilePath, 'image/png');

    const url = `${AWS_S3_AD_VIDEOS_CLOUD_FRONT_URL}/${videoFilePath}`;
    const lowUrl = `${AWS_S3_AD_VIDEOS_CLOUD_FRONT_URL}/${videoLowFilePath}`;
    const coverUrl = `${AWS_S3_AD_VIDEOS_CLOUD_FRONT_URL}/${coverVideoFilePath}`;

    console.log('Video URL:', url);
    console.log('Low Video URL:', lowUrl);
    console.log('Cover URL:', coverUrl);

    await axios.patch(callBackUrl, {
      url,
      lowUrl,
      coverUrl,
      status: 'done'
    })

    return true;
  } catch (error) {
    console.error(adVideoId, userId)
    console.error(error);
    await axios.patch(errorCallBackUrl, {
      error,
      status: 'error'
    })
  } finally {
    fs.rm(folderPath, { recursive: true }, (err) => {
      if (err) {
        console.error(err);
        return;
      }
    });
  }
}



async function prepareMedia(folderPath, data) {
  console.log('Preparing media');
  const {
    media
  } = data;

  const videos = media.filter(mediaItem => mediaItem.type === 'video');
  const images = media.filter(mediaItem => mediaItem.type === 'image');
  
  const processImage = async (mediaItem, index) => {
    const mediaPathOrg = `${folderPath}/${index + 1}_org.png`;
    const mediaPath = `${folderPath}/${index + 1}.png`;
    try {
      await download(mediaItem.url, mediaPathOrg);
      await sharp(mediaPathOrg)
        .rotate()
        .resize(1216)
        .jpeg({ quality: 100 })
        .toFile(mediaPath);
      fs.unlinkSync(mediaPathOrg);
      return { success: true, path: mediaPath };
    } catch (error) {
      console.error('Error processing image:', mediaItem.url);
      return { success: false, path: mediaPath };
    }
  };

  const results = await Promise.all(images.map(processImage));
  const goodImages = results.filter(result => result.success).map(result => result.path);
  const badImages = results.filter(result => !result.success).map(result => result.path);

  if (goodImages.length === 0) {
    throw new Error('No images to process');
  }

  if (badImages.length > 0) {
    let i = 0;
    while (badImages.length > 0) {
      const goodImage = goodImages[i];
      const badImage = badImages.shift();
      fs.copyFileSync(goodImage, badImage);
      i++;
      if (i >= goodImages.length) {
        i = 0;
      }
    }
  }
  
  await Promise.all(
    videos.map(async (mediaItem, index) => {
      const mediaPath = `${folderPath}/${index+1}.mp4`;
      await download(mediaItem.url, mediaPath);
    })
  );
  
  console.log('Media prepared');
  
  return true;
}

async function prepare(folderPath, data) {
  const {
    media,
    avatarUrl,
    actionUrl,
    musicUrl,
    avatarSettings,
    subtitleUrl,
    textHookImageUrl
  } = data;
  const avatarSubtitlesUrl = subtitleUrl ? subtitleUrl : avatarUrl.replace('.mp4', '.srt');

  const subtitlesFormat = avatarSubtitlesUrl.includes('.srt') ? `srt` : `ass`

  if(textHookImageUrl) download(textHookImageUrl, `${folderPath}/textHookImage.png`),
  
  await Promise.all([
    download(avatarUrl, `${folderPath}/avatar.mp4`),
    download(actionUrl, `${folderPath}/action.mp4`),
    download(musicUrl, `${folderPath}/background_audio.mp3`),
    download(avatarSettings.bgImageUrl, `${folderPath}/background_image.jpg`),
    download(avatarSubtitlesUrl, `${folderPath}/subtitles.${subtitlesFormat}`),
  ]);

  let acceleration= 1

  if(avatarSettings && avatarSettings.accelerationEnabled) {
    acceleration= await getAcceleration(`${folderPath}/avatar.mp4`, 27);
    await runCommand(`ffmpeg -i ${folderPath}/avatar.mp4  -vf "setpts=${1/acceleration}*PTS" -filter:a "atempo=${acceleration}" -q:v 3 -q:a 3 ${folderPath}/avatar_speed.mp4`);
    await runCommand(`ffmpeg -i ${folderPath}/avatar_speed.mp4 -filter:a "volume=2.0" ${folderPath}/avatar_end.mp4`);
    fs.unlinkSync(`${folderPath}/avatar_speed.mp4`);
  } else {
    await runCommand(`ffmpeg -i ${folderPath}/avatar.mp4 -filter:a "volume=2.0" ${folderPath}/avatar_end.mp4`);
  }

  fs.renameSync(`${folderPath}/avatar.mp4`, `${folderPath}/avatar_org.mp4`);

  await runCommand(`ffmpeg -i ${folderPath}/avatar_end.mp4 -vf "scale=1216:2160" ${folderPath}/avatar.mp4`);

  if(subtitlesFormat === 'srt') {
    await generateSubtitlesSRT({
      folderPath,
      acceleration
    })
  } else {
    await generateSubtitlesASS({
      folderPath,
      acceleration
    })
  }

  await Promise.all([
    runCommand(`./templates/common/run.sh ${folderPath}`),
  ]);
}

async function generateSubtitles(folderPath) {
  await runCommand(`whisper_timestamped ${folderPath}/avatar.mp4 --model small --output_dir ${folderPath}/words`);
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

async function generateSubtitlesVoice(folderPath) {
  await runCommand(`whisper_timestamped ${folderPath}/voice.mp3 --model small --output_dir ${folderPath}/words`);
  const data = fs.readFileSync(`${folderPath}/words/voice.mp3.words.json`, 'utf8');
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

async function prepareVoice(folderPath, data) {
  const {
    voiceUrl,
    actionUrl,
    musicUrl,
  } = data;
  
  await Promise.all([
    download(voiceUrl, `${folderPath}/voice.mp3`),
    download(actionUrl, `${folderPath}/action.mp4`),
    download(musicUrl, `${folderPath}/background_audio.mp3`),
  ]);

  await generateSubtitlesVoice(folderPath);
}

async function handleVoice(templateName, data) {
  const {
    adVideoId,
    userId,
    callBackUrl,
    errorCallBackUrl,
    productUrl,
    subtitleTemplate = ''
  } = data;
  console.log('Processing videos', adVideoId, userId)
  console.time('BUILD_TIME');
  //TODO: Add error handling
  const folderPath = `./data/${uuidv4()}`;
  if (fs.existsSync(folderPath)) {
    fs.rmSync(`${folderPath}`, { recursive: true }, (err) => {
      if (err) {
        console.error(err);
        return;
      }
    });
  }
  fs.mkdirSync(folderPath, { recursive: true });
  try {
    
    await prepareMedia(folderPath, data);

    await Promise.all([
      runSiteScroll(productUrl, `${folderPath}/ss.mp4`, folderPath, 25, true),
      prepareVoice(folderPath, data)
    ]);

    await runCommand(`./templates/${templateName}/run.sh ${folderPath} ${subtitleTemplate}`);
    
    const fileName = uuidv4();
    
    await runCommand(`ffmpeg -i ${folderPath}/output.mp4 -vf "scale=1152:2048" ${folderPath}/output_scale_command.mp4`);
    await runCommand(`ffmpeg -i ${folderPath}/output.mp4 -vf "scale=720:1280" ${folderPath}/output_low_scale_command.mp4`);
    
    fs.unlinkSync(`${folderPath}/output.mp4`);
    fs.renameSync(`${folderPath}/output_scale_command.mp4`, `${folderPath}/output.mp4`);
    
    const coverPath = await createCover(`${folderPath}/output.mp4`, folderPath);
    
    console.timeEnd('BUILD_TIME');

    const videoFilePath = `${userId}/${adVideoId}/${fileName}.mp4`
    const videoLowFilePath = `${userId}/${adVideoId}/${fileName}_low.mp4`
    const coverVideoFilePath = `${userId}/${adVideoId}/${fileName}_cover.png`
    
    await uploadToS3(`${folderPath}/output.mp4`, videoFilePath);
    await uploadToS3(`${folderPath}/output_low_scale_command.mp4`, videoLowFilePath);
    await uploadToS3(coverPath, coverVideoFilePath, 'image/png');

    const url = `${AWS_S3_AD_VIDEOS_CLOUD_FRONT_URL}/${videoFilePath}`;
    const lowUrl = `${AWS_S3_AD_VIDEOS_CLOUD_FRONT_URL}/${videoLowFilePath}`;
    const coverUrl = `${AWS_S3_AD_VIDEOS_CLOUD_FRONT_URL}/${coverVideoFilePath}`;

    console.log('Video URL:', url);
    console.log('Low Video URL:', lowUrl);
    console.log('Cover URL:', coverUrl);

    await axios.patch(callBackUrl, {
      url,
      lowUrl,
      coverUrl,
      status: 'done'
    })

    return true;
  } catch (error) {
    console.error(adVideoId, userId)
    console.error(error);
    await axios.patch(errorCallBackUrl, {
      error,
      status: 'error'
    })
  } finally {
    fs.rm(folderPath, { recursive: true }, (err) => {
      if (err) {
        console.error(err);
        return;
      }
    });
  }
}

async function createCover(videoPath, folderPath) {
  const outputPath = `${folderPath}/output_cover.jpeg`;
  await runCommand(`ffmpeg -ss 1 -i ${videoPath} -vframes 1 -compression_level 6 -q:v 80 ${outputPath}`);
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

async function getVideoDuration(filePath) {
  const data = await runCommand(`ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 ${filePath}`);
  return parseFloat(data.trim());
}

async function getAcceleration(inputPath, minDuration) {
  const standardAcceleration = 1.3;
  try {
      const duration = await getVideoDuration(inputPath);
      console.log(`Video duration: ${duration}`);
      let acceleration = standardAcceleration;

      let resultingDuration = duration / acceleration;

      if (resultingDuration < minDuration) {
          acceleration = duration / minDuration;
          if(acceleration < 1) {
            acceleration = 1;
          }
      }
      console.log(`Video accelerated with factor ${acceleration}.`);
      
      return acceleration;
  } catch (error) {
      console.error('Error accelerating video:', error);
  }
}



module.exports = {
  handle,
  handleVoice
}