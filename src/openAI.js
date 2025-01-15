const axios = require('axios');
const AWS = require('aws-sdk');
const fs = require('fs');
const OpenAI = require('openai');

const {
  AWS_KEY_ID, 
  AWS_SECRET_KEY, 
  AWS_S3_AD_VIDEOS_BUCKET, 
  OPENAI_API_KEY
} = process.env;

AWS.config.update({
  accessKeyId: AWS_KEY_ID,
  secretAccessKey: AWS_SECRET_KEY
});

const client = new OpenAI({
  apiKey: process.env['OPENAI_API_KEY'],
});

const s3 = new AWS.S3();

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

function removeFromS3(s3Key) {
  const params = {
    Bucket: AWS_S3_AD_VIDEOS_BUCKET,
    Key: s3Key,
  };
  return new Promise((resolve, reject) => {
    s3.deleteObject(params, function(err, data) {
      if (err) {
        reject(err);
      }
      console.log(`File removed successfully. ${data.Location}`);
      resolve(data.Location);
    });
  });
}

const parseAnalysisResult = (result) => {
  try {
      const match = result.match(/VALID:\s*(true|false)/i);

      if (match) {
          return match[1].toLowerCase() === 'true';
      }

      throw new Error('VALID: true/false not found in the response');
  } catch (error) {
      console.error('Error parsing analysis result:', error.message);
      return null;
  }
};

const analyzeScreenshot = async (filePath) => {
  const key = `screenshot-${Date.now()}.png`;
  try {
      const imageUrl = await uploadToS3(filePath, key, 'image/png');
      
      const response = await client.chat.completions.create({
        model: 'gpt-4o-mini',
        messages: [
            {
                role: 'system',
                content: 'You are an assistant analyzing screenshots of web pages. Your job is to determine if the webpage loaded correctly or if there is any error visible on the screen. Be specific and mention any issues you observe.'
            },
            {
                role: 'user',
                content: [
                  {
                    type: 'text',
                    text: 'Here is a screenshot of the webpage. Please analyze it and tell me if the page appears to be loaded correctly or if there are any errors displayed on the screen. Return response in form VALID: true/false',
                  },
                  {
                    type: 'image_url',
                    image_url: {
                        url: imageUrl,
                        detail: 'auto'
                    },
                },
                ],
            },
        ],
      });
      const result = response.choices[0].message.content;
      
      const isValid = parseAnalysisResult(result);
      
      console.log('Is page valid:', isValid);
      
      await removeFromS3(key);
      
      return isValid;
  } catch (error) {
      console.error('Error analyzing screenshot:', error.message);
      await removeFromS3(key);
      return null;
  } 
};

module.exports = {
  analyzeScreenshot
}