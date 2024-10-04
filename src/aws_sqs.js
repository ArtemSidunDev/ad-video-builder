const AWS = require('aws-sdk');
const handler = require('./handlers');
const fs = require('fs');

const { AWS_KEY_ID, AWS_SECRET_KEY, AWS_SQS_URL } = process.env;

AWS.config.update({
  accessKeyId: AWS_KEY_ID,
  secretAccessKey: AWS_SECRET_KEY
});

const sqs = new AWS.SQS();

const receiveParams = {
  QueueUrl:            AWS_SQS_URL,
  MaxNumberOfMessages: 1, 
  VisibilityTimeout:   600, // 10 minutes
  WaitTimeSeconds:     20, // 20 seconds
};

const pollMessages = async () => {
  console.log(new Date(), 'Polling messages');
  const checkResult = await checkProcessMessages();
  console.log('Check result:', checkResult);
  if(checkResult) {
    sqs.receiveMessage(receiveParams, (err, data) => {
      if (err) {
        console.error('Error receiving message:', err);
      }
      
      console.log(new Date(), 'Received message:', data.Messages.length);

      if (data.Messages && data.Messages.length) {
        for (const message of data.Messages) {
          processMessage(message);
        }
      }
    });
  }

  setTimeout(pollMessages, 20000); //20 seconds
};

async function processMessage(message) {
  try {
    const body = JSON.parse(message.Body);
    const templateName = body.templateName;
    const data = body.data;

    await handler.handle(templateName, data);

    const deleteParams = {
      QueueUrl:      AWS_SQS_URL,
      ReceiptHandle: message.ReceiptHandle,
    };

    sqs.deleteMessage(deleteParams, (err, data) => {
      if (err) {
        console.error('Error delete message:', err);
      } else {
        console.log('Message deleted:', data);
      }
    });
  }
  catch (error) {
    console.error('Error processing message FROM SQS*******************************:', error);
    console.error(error);
  }
}

function getFolderCount(directoryPath) {
  return new Promise((resolve, reject) => {
    fs.readdir(directoryPath, { withFileTypes: true }, (err, files) => {
      if (err) {
        reject(err);
        return;
      }
      const folders = files.filter(file => file.isDirectory());
      resolve(folders.length);
    });
  });
}

async function checkProcessMessages() {
  const dataFolder = './data';

  const count = await getFolderCount(dataFolder)

  return count < 3;
}

module.exports = {
  pollMessages
};