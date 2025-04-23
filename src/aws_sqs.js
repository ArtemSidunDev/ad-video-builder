const AWS = require('aws-sdk');
const handler = require('./handlers');
const fs = require('fs');
const { v4: uuidv4 } = require('uuid');
const axios = require('axios');

const { 
  AWS_KEY_ID, 
  AWS_SECRET_KEY, 
  AWS_SQS_URL,
  TELEGRAM_BOT_TOKEN,
  TELEGRAM_CHAT_ID,
} = process.env;
const maxProcesses = 2;

AWS.config.update({
  accessKeyId: AWS_KEY_ID,
  secretAccessKey: AWS_SECRET_KEY
});

const sqs = new AWS.SQS();

const foldersMap = {};

const receiveParams = {
  QueueUrl:            AWS_SQS_URL,
  MaxNumberOfMessages: 1,
  VisibilityTimeout:   60 * 20, // 20 minutes
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
  const deleteParams = {
    QueueUrl:      AWS_SQS_URL,
    ReceiptHandle: message.ReceiptHandle,
  };
  try {
    const body = JSON.parse(message.Body);
    const templateName = body.templateName;
    const data = body.data;
    const folderId= uuidv4();
    foldersMap[folderId] = {
      createdAt: Date.now(),
      folderPath: `./data/${folderId}`,
      data: data,
      templateName: templateName
    }

    console.log('Processing message:', message.MessageId, 'Data Avatar Id:', data.adVideoId, 'Template name:', templateName, 'Folder id:', folderId, );
    await handler.handle(templateName, data, `./data/${folderId}`);

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
    sqs.deleteMessage(deleteParams, (err, data) => {
      if (err) {
        console.error('Error delete message:', err);
      } else {
        console.log('Message deleted:', data);
      }
    });
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

async function sendTgMessage(data, templateName, createdAt) {
  const date = new Date(createdAt);
  const year = date.getFullYear();
  const month = String(date.getMonth() + 1).padStart(2, '0');
  const day = String(date.getDate()).padStart(2, '0');
  const hours = String(date.getHours()).padStart(2, '0');
  const minutes = String(date.getMinutes()).padStart(2, '0');
  const seconds = String(date.getSeconds()).padStart(2, '0');
  const createdAtFormat = `${year}-${month}-${day} ${hours}:${minutes}:${seconds}`;
  
  const message = `<b>AD VIDEO BUILDER ERROR</b>
  %0A - User Id: <code>${data ? data.userId : 'N/A'}</code>
  %0A - AdVideo Id: <code>${data ? data.adVideoId : 'N/A'}</code>
  %0A - Template name: <code>${templateName || 'N/A'}</code>
  %0A - Created At: <code>${createdAtFormat || 'N/A'}</code>
  %0A - Error: <code>Video folder deleted by time</code>`;
  return await axios.post(`https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/sendMessage?chat_id=${TELEGRAM_CHAT_ID}&parse_mode=HTML&text=${message}`);
}

async function checkFoldersAndDelete(directoryPath) {
  fs.readdir(directoryPath, { withFileTypes: true }, (err, files) => {
    if (err) {
      console.error('Error reading directory:', err);
      return;
    }

    files.forEach(file => {
      if (file.isDirectory()) {
        const folderPath = `${directoryPath}/${file.name}`;
        if(!foldersMap[file.name]) {
          foldersMap[file.name] = {
            createdAt: Date.now(),
            folderPath: folderPath
          }
        } else {
          const folder = foldersMap[file.name];
          const currentTime = Date.now();
          const timeDiff = currentTime - folder.createdAt;
          const timeLimit = 1000 * 60 * 30 // 30 minutes

          console.log('Folder:', folderPath, 'Created at:', new Date(folder.createdAt), 'Current time:', new Date(currentTime), 'Time diff in minutes:', Math.floor(timeDiff / (1000 * 60)));

          if (timeDiff > timeLimit) {
            fs.rm(folder.folderPath, { recursive: true }, (err) => {
              if (err) {
                console.error('Error deleting folder:', err);
              } else {
                sendTgMessage(folder.data, folder.templateName, folder.createdAt)
                console.log('Deleted folder:', folder.folderPath);

                delete foldersMap[file.name];
              }
            });
          }
        }
      }
    });
  });
}

async function checkProcessMessages() {
  const dataFolder = './data';

  const count = await getFolderCount(dataFolder)
  if(count > 0) {
    console.log('Count of folders:', count);
    await checkFoldersAndDelete(dataFolder);
  }

  return count;
}

module.exports = {
  pollMessages
};