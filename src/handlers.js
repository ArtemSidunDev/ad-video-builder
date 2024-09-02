const { exec } = require('child_process');
const axios = require('axios');
const fs = require('fs');
const AWS = require('aws-sdk');
const sharp = require('sharp');
const { v4: uuidv4 } = require('uuid');
const { run: runSiteScroll } = require('./site_scroll.js');
const { run: generateSubtitlesGPT } = require('./subtitles_gpt.js');
const { run: generateSubtitlesSRT } = require('./subtitles_srt.js');
const { run: generateSubtitlesAzure } = require('./subtitles_azure.js');

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
    productUrl,
    subtitleTemplate = ''
  } = data;
  console.time('BUILD_TIME');
  
  const folderPath = `./data/${adVideoId}`;
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
    await Promise.all([
      runSiteScroll(productUrl, `${folderPath}/ss.mp4`, folderPath, 25),
      prepare(folderPath, data)
    ]);

    await runCommand(`./templates/${templateName}/run.sh ${folderPath} ${subtitleTemplate}`);
    
    const fileName = uuidv4();
    
    await runCommand(`ffmpeg -i ${folderPath}/output.mp4 -vf "scale=2160:3840" ${folderPath}/output_scale_command.mp4`);
    
    fs.unlinkSync(`${folderPath}/output.mp4`);
    fs.renameSync(`${folderPath}/output_scale_command.mp4`, `${folderPath}/output.mp4`);
    
    const coverPath = await createCover(`${folderPath}/output.mp4`, folderPath);
    
    console.timeEnd('BUILD_TIME');

    const url = await uploadToS3(`${folderPath}/output.mp4`, `${userId}/${adVideoId}/${fileName}.mp4`);
    const coverUrl = await uploadToS3(coverPath, `${userId}/${adVideoId}/${fileName}_cover.png`, 'image/png');

    await axios.patch(callBackUrl, {
      url,
      coverUrl,
      status: 'done'
    })

    return true;
  } catch (error) {
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

async function prepare(folderPath, data) {
  const {
    media,
    avatarUrl,
    actionUrl,
    musicUrl,
    script
  } = data;
  const avatarSubtitlesUrl = avatarUrl.replace('.mp4', '.srt');
  const videos = media.filter(mediaItem => mediaItem.type === 'video');
  const images = media.filter(mediaItem => mediaItem.type === 'image');
  
  await Promise.all([
    download(avatarUrl, `${folderPath}/avatar.mp4`),
    download(actionUrl, `${folderPath}/action.mp4`),
    download(musicUrl, `${folderPath}/background_audio.mp3`),
    download(avatarSubtitlesUrl, `${folderPath}/subtitles.srt`),
    videos.map(async (mediaItem, index) => {
      const mediaPath = `${folderPath}/${index+1}.mp4`;
      await download(mediaItem.url, mediaPath);
    }),
    images.map(async (mediaItem, index) => {
      const mediaPathOrg = `${folderPath}/${index+1}_org.png`;
      const mediaPath = `${folderPath}/${index+1}.png`;
      await download(mediaItem.url, mediaPathOrg);
  
      await sharp(mediaPathOrg)
      .resize(2432)
      .jpeg({ quality: 100 })
      .toFile(mediaPath);
      fs.unlinkSync(mediaPathOrg);
    })
  ]);
  const acceleration= await getAcceleration(`${folderPath}/avatar.mp4`, 27);

  await runCommand(`ffmpeg -i ${folderPath}/avatar.mp4  -vf "setpts=${1/acceleration}*PTS" -filter:a "atempo=${acceleration}" -q:v 3 -q:a 3 ${folderPath}/avatar_speed.mp4`);
  await runCommand(`ffmpeg -i ${folderPath}/avatar_speed.mp4 -filter:a "volume=2.0" ${folderPath}/avatar_end.mp4`);

  fs.unlinkSync(`${folderPath}/avatar_speed.mp4`);
  
  fs.renameSync(`${folderPath}/avatar.mp4`, `${folderPath}/avatar_org.mp4`);
  fs.renameSync(`${folderPath}/avatar_end.mp4`, `${folderPath}/avatar.mp4`);

  await runCommand(`./templates/common/run.sh ${folderPath}`);
  
  // await generateSubtitlesAzure({
  //   folderPath,
  //   script
  // });

  // await generateSubtitlesGPT({
  //   folderPath,
  //   script
  // })

  await generateSubtitlesSRT({
    folderPath,
    acceleration
  })
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
  handle
}