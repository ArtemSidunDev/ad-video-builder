// start express server
const dotenv = require('dotenv');
dotenv.config('./.env');

const handler = require('./handler');
const express = require('express');
const app = express();

const port = process.env.PORT

app.use(express.json());

app.get('/', (req, res) => {
  res.send('Hello World!');
});

app.post('/:templateName', async (req, res) => {
  try {
    const templateName = req.params.templateName;
    const data = await handler.handle(templateName, req.body);
    res.send({
      message: 'Processing request',
      data
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

app.listen(port, () => {
  console.log(`app listening at ${port}`);
});