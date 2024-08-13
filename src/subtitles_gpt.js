const fs = require('fs');
const axios = require('axios');
const { exec } = require('child_process');

const { OPENAI_API_KEY } = process.env;

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

async function getCorrectedSubtitles(originalText, subtitles) {
  const prompt = `
  Given the original text and the subtitles JSON, correct the subtitles to match the original text perfectly. Ensure there are no errors and return only the corrected subtitles in JSON format.
  
  Don't change the order of the subtitles or the timestamps. If a subtitle text is incorrect, replace it with the correct text. If a subtitle is missing, add it with the correct text and timestamps. If there are extra subtitles, remove them.

  Original Text:
  ${originalText}

  Subtitles:
  ${JSON.stringify(subtitles)}

  Corrected Subtitles (JSON only):
  `;

  try {
      const response = await axios.post('https://api.openai.com/v1/chat/completions', {
          model: 'gpt-4',
          messages: [
              { role: 'system', content: 'You are a helpful assistant.' },
              { role: 'user', content: prompt }
          ],
          max_tokens: 4096,
          temperature: 0
      }, {
          headers: {
              'Authorization': `Bearer ${OPENAI_API_KEY}`,
              'Content-Type': 'application/json'
          }
      });
      console.log(JSON.stringify(response.data, null, 2));
      const correctedSubtitles = response.data.choices[0].message.content;
      return correctedSubtitles;

  } catch (error) {
      console.error('Error while calling ChatGPT API:', error);
      throw error;
  }
}

async function generateSubtitles(folderPath, script) {
  await runCommand(`whisper_timestamped ${folderPath}/avatar.mp4 --model large-v3 --language en --output_dir ${folderPath}/words`);
  const data = fs.readFileSync(`${folderPath}/words/avatar.mp4.words.json`, 'utf8');
  const parsedData = JSON.parse(data);
  let words = [];
  for(const segment of parsedData.segments) {
    for(const word of segment.words) {
      words.push({
        start: word.start,
        end:   word.end,
        word:  word.text
      });
    }
  }
  fs.writeFileSync(`${folderPath}/transcriptionw.json`, JSON.stringify(words), { encoding: 'utf8' });
  // try {
  //   const correctedSubtitles = await getCorrectedSubtitles(script, words);
  //   words = JSON.parse(correctedSubtitles);
  // } catch (error) {
  //   console.error('Error while calling ChatGPT API:', error);
  // }

  fs.writeFileSync(`${folderPath}/transcription.json`, JSON.stringify(words), { encoding: 'utf8' });
  fs.rm(`${folderPath}/words`, { recursive: true }, (err) => {
    if (err) {
      console.error(err);
      return;
    }
  });
  return `${folderPath}/transcription.json`;
}

const run = async ({ folderPath, script}) => {
  const transcriptionPath = await generateSubtitles(folderPath, script);
  return transcriptionPath;
}

module.exports = {
  run
}