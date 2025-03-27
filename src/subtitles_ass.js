const fs = require('fs').promises;
const path = require('path');
const he = require('he');

function applyAcceleration(timeStr, acceleration) {
  const [hours, minutes, seconds] = timeStr.split(/:|\./).map(Number);
  let totalMilliseconds = hours * 3600000 + minutes * 60000 + Math.floor(seconds * 1000);
  totalMilliseconds /= acceleration;

  const newHours = Math.floor(totalMilliseconds / 3600000);
  totalMilliseconds %= 3600000;
  const newMinutes = Math.floor(totalMilliseconds / 60000);
  totalMilliseconds %= 60000;
  const newSeconds = Math.floor(totalMilliseconds / 1000);
  const newMilliseconds = Math.floor(totalMilliseconds % 1000);

  return `${String(newHours).padStart(2, '0')}:${String(newMinutes).padStart(2, '0')}:${String(newSeconds).padStart(2, '0')}.${String(newMilliseconds).padStart(3, '0')}`;
}

async function changeSubtitleTimingASS(filePath, outputPath, acceleration) {
  try {
    const fileContent = await fs.readFile(filePath, 'utf8');
    const lines = fileContent.split('\n');
    const newLines = lines.map(line => {
      if (line.startsWith('Dialogue:')) {
        const parts = line.split(',');
        if (parts.length > 2) {
          parts[1] = applyAcceleration(parts[1], acceleration); // Начальное время
          parts[2] = applyAcceleration(parts[2], acceleration); // Конечное время
          return parts.join(',');
        }
      }
      return line;
    });
    await fs.writeFile(outputPath, newLines.join('\n'), 'utf8');
    console.log('ASS subtitles timing adjusted.');
  } catch (error) {
    console.error(`Error processing ASS subtitles: ${error.message}`);
  }
}

function parseASS(data) {
  const result = [];
  const assDialoguesRegex = /^Dialogue: [^,]*,(\d+:\d+:\d+\.\d+),(\d+:\d+:\d+\.\d+),[^,]*,[^,]*,[^,]*,[^,]*,[^,]*,[^,]*,(.*)$/gm;
  let match;

  while ((match = assDialoguesRegex.exec(data)) !== null) {
    const text = match[3]
      .replace(/{.*?}/g, '')
      .replace(/\\N/g, ' ')
      .replace(/\n/g, '')
      .replace(/—/g, '-')
      .replace(/\s+([.,!?])/g, '$1')
      .replace(/([.,])(?=\S)/g, '$1 ')
      .replace(/(\s*)\\n(\s*)/g, ' ')
      .trim();

    result.push({
      start: convertAssTimeToSeconds(match[1]),
      end: convertAssTimeToSeconds(match[2]),
      text: text
    });
  }
  return result;
}

function convertAssTimeToSeconds(timeStr) {
  const [hours, minutes, seconds] = timeStr.split(/:|\./).map(Number);
  return hours * 3600 + minutes * 60 + seconds;
}

function calculateWordTimings(subtitles) {
  return subtitles.flatMap(subtitle => {
    const words = subtitle.text.split(' ');
    const totalDuration = subtitle.end - subtitle.start;

    // Подсчёт букв без дефисов
    const lettersPerWord = words.map(word => word.replace(/-/g, '').length);
    const totalLetters = lettersPerWord.reduce((sum, len) => sum + len, 0);
    
    if (totalLetters === 0) return []; // Предотвращение деления на 0

    let currentStart = subtitle.start;
    return words.map((word, index) => {
      const wordDuration = lettersPerWord[index] / totalLetters * totalDuration;
      const wordEnd = currentStart + wordDuration;

      const wordTiming = {
        start: parseFloat(currentStart.toFixed(3)),
        end: parseFloat(wordEnd.toFixed(3)),
        word: he.decode(word)
      };

      currentStart = wordEnd;
      return wordTiming;
    });
  });
}

async function readFileAndParseASS(filePath) {
  try {
    const data = await fs.readFile(filePath, 'utf8');
    return parseASS(data);
  } catch (error) {
    console.error(`Error reading or parsing ASS file: ${error.message}`);
    return [];
  }
}

const run = async ({ folderPath, acceleration }) => {
  try {
    const inputFilePath = path.join(folderPath, 'subtitles.ass');
    const outputFilePath = path.join(folderPath, 'subtitles_speed.ass');

    await changeSubtitleTimingASS(inputFilePath, outputFilePath, acceleration);

    const jsonArray = await readFileAndParseASS(outputFilePath);
    if (jsonArray.length > 0) {
      const words = calculateWordTimings(jsonArray);
      const outputJsonPath = path.join(folderPath, 'transcription.json');
      await fs.writeFile(outputJsonPath, JSON.stringify(words, null, 2), 'utf8');
      console.log('Transcription and word timings saved.');
    } else {
      console.log('No subtitles found or failed to parse.');
    }
  } catch (error) {
    console.error(`Error in run process for ASS: ${error.message}`);
  }
};

module.exports = {
  run
};