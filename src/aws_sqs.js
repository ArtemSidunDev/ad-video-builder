const AWS = require('aws-sdk');
const handler = require('./handlers');
const fs = require('fs');

const { AWS_KEY_ID, AWS_SECRET_KEY, AWS_SQS_URL } = process.env;
const maxProcesses = 2;

AWS.config.update({
  accessKeyId: AWS_KEY_ID,
  secretAccessKey: AWS_SECRET_KEY
});

const sqs = new AWS.SQS();

const receiveParams = {
  QueueUrl:            AWS_SQS_URL,
  MaxNumberOfMessages: 1,
  VisibilityTimeout:   60 * 4, // 40 minutes
  WaitTimeSeconds:     10, // 20 seconds
};

const pollMessages = async () => {
  console.log(new Date(), 'Start polling messages process');
  
  const checkResult = await checkProcessMessages();
  
  console.log('Check result:', checkResult);
  
  if(checkResult < maxProcesses) {
    
    receiveParams.MaxNumberOfMessages = maxProcesses - checkResult;
    
    console.log(new Date(), 'Polling messages');
    
    sqs.receiveMessage(receiveParams, (err, data) => {
      if (err) {
        console.error('Error receiving message:', err);
      }
      
      console.log(new Date(), 'Received message:', data.Messages.length);

      if (data.Messages && data.Messages.length) {
        for (const message of data.Messages) {
          console.log('Processing message:', message.MessageId);
          processMessage(message);
        }
      }

      console.log('Waiting for next poll');
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

  return count;
}

module.exports = {
  pollMessages
};