// start express server
const dotenv = require('dotenv');
dotenv.config('./.env');

const handler = require('./src/handlers');
const awsSqs = require('./src/aws_sqs');
const express = require('express');
const app = express();
const fs = require('fs');

awsSqs.pollMessages();

const port = process.env.PORT

app.use(express.json());

app.get('/', (req, res) => {
  res.send('Hello World!');
});

app.post('/:templateName', async (req, res) => {
  try {
    const templateName = req.params.templateName;
    
    handler.handle(templateName, req.body);
    
    res.send({
      message: 'Processing request',
      status: 'ok'
    });
  }
  catch (error) {
    console.log(error);
    res.send({
      message: 'Processing request error',
      error: error
    });
  }
});

fs.readdirSync('./data').forEach(file => {
  if (fs.lstatSync(`./data/${file}`).isDirectory()) {
    fs.rmdirSync(`./data/${file}`, { recursive: true });
  }
})

app.listen(port, () => {
  console.log(`app listening at ${port}`);
});