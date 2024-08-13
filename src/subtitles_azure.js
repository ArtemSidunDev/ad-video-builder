const fs = require('fs');
const path = require('path');
const { exec } = require('child_process');
const sdk = require("microsoft-cognitiveservices-speech-sdk");

const { AZURE_SPEECH_KEY, AZURE_SERVICE_REGION } = process.env;

async function runCommand(command) {
  console.log('\n');
  console.log(command);
  console.log('\n');
  return new Promise((resolve, reject) => {
    const process = exec(command, (error, stdout, stderr) => {
      if (error) {
        console.error(`exec error: ${error}`);
        console.error(`exec stderr: ${stderr}`);
        reject(error);
        return;
      }
      resolve(stdout);
    });

    process.stdout.on('data', (data) => {
      console.log('stdout: ' + data);
    });

    process.stderr.on('data', (data) => {
      console.log('stderr: ' + data);
    });
  });
}

async function prepare(folderPath, script) {
  console.log('Preparing transcription...');
  console.log('Folder path:', AZURE_SPEECH_KEY, AZURE_SERVICE_REGION);
  const speechConfig = sdk.SpeechConfig.fromSubscription(AZURE_SPEECH_KEY, AZURE_SERVICE_REGION);
  speechConfig.outputFormat = sdk.OutputFormat.Detailed;
  speechConfig.requestWordLevelTimestamps();

  const audioConfig = sdk.AudioConfig.fromWavFileInput(fs.readFileSync(`${folderPath}/avatar.wav`));

  const speechRecognizer = new sdk.SpeechRecognizer(speechConfig, audioConfig);

  const phraseListGrammar = sdk.PhraseListGrammar.fromRecognizer(speechRecognizer);

  const phraseList = script.split(' ');

  phraseList.forEach(phrase => {
    phraseListGrammar.addPhrase(phrase);
  });

  const wordsWithTimestamps = [];

  return new Promise((resolve, reject) => {
    // Событие для обработки промежуточных результатов
    speechRecognizer.recognizing = (s, e) => {
      console.log(`RECOGNIZING: Text=${e.result.text}`);
    };

    // Событие для обработки окончательных результатов
    speechRecognizer.recognized = (s, e) => {
      if (e.result.reason === sdk.ResultReason.RecognizedSpeech) {
        console.log(`RECOGNIZED: Text=${e.result.text}`);

        // Получение подробных результатов
        const detailedResults = JSON.parse(e.result.json);
        if (detailedResults.NBest && detailedResults.NBest[0].Words) {
          detailedResults.NBest[0].Words.forEach(wordInfo => {
            const startSeconds = wordInfo.Offset / 10000000;
            const durationSeconds = wordInfo.Duration / 10000000;
            wordInfo.Word = wordInfo.Word.replace(/,dot,/g, '.');
            wordsWithTimestamps.push({
              word: wordInfo.Word,
              start: startSeconds,
              end: startSeconds + durationSeconds
            });
          });
        }
      }
    };

    // Событие для обработки отмены
    speechRecognizer.canceled = (s, e) => {
      console.log(`CANCELED: Reason=${e.reason}`);
      if (e.reason === sdk.CancellationReason.Error) {
        console.log(`CANCELED: ErrorDetails=${e.errorDetails}`);
        reject(e.errorDetails);
      }
      speechRecognizer.stopContinuousRecognitionAsync();
    };

    // Событие для обработки завершения распознавания
    speechRecognizer.sessionStopped = (s, e) => {
      console.log("\nSession stopped event.");

      // Сохранение результата в JSON файл
      if (wordsWithTimestamps.length > 0) {
        const wordsWithTimestampsPath = path.join(folderPath, 'transcription.json');
        fs.writeFileSync(wordsWithTimestampsPath, JSON.stringify(wordsWithTimestamps, null, 2));
        console.log("Words with timestamps saved to output_words.json");
        resolve(wordsWithTimestampsPath);
      }

      speechRecognizer.stopContinuousRecognitionAsync();
    };

    // Запуск непрерывного распознавания
    speechRecognizer.startContinuousRecognitionAsync();
  });
}

const run = async ({ folderPath, script }) => {
  const avatarVideoPath = path.join(folderPath, 'avatar.mp4');
  const avatarAudioPath = path.join(folderPath, 'avatar.wav');
  await runCommand(`ffmpeg -i ${avatarVideoPath} -vn -acodec pcm_s16le -ar 44100 -ac 2 ${avatarAudioPath}`);

  const result = await prepare(folderPath, script);

  return result;
};

module.exports = {
  run
};