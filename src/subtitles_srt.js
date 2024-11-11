const fs = require('fs').promises;
const path = require('path');
const he = require('he');

function applyAcceleration(timeStr, acceleration) {
  const [hours, minutes, seconds, milliseconds] = timeStr.split(/[:,]/).map(Number);
  let totalMilliseconds = hours * 3600000 + minutes * 60000 + seconds * 1000 + milliseconds;
  totalMilliseconds /= acceleration;

  const newHours = Math.floor(totalMilliseconds / 3600000);
  totalMilliseconds %= 3600000;
  const newMinutes = Math.floor(totalMilliseconds / 60000);
  totalMilliseconds %= 60000;
  const newSeconds = Math.floor(totalMilliseconds / 1000);
  const newMilliseconds = Math.floor(totalMilliseconds % 1000);

  return `${String(newHours).padStart(2, '0')}:${String(newMinutes).padStart(2, '0')}:${String(newSeconds).padStart(2, '0')},${String(newMilliseconds).padStart(3, '0')}`;
}

async function changeSubtitleTiming(filePath, outputPath, acceleration) {
  try {
    const fileContent = await fs.readFile(filePath, 'utf8');
    const lines = fileContent.split('\n');
    const newLines = lines.map(line => {
      const match = line.match(/(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})/);
      if (match) {
        const startTime = applyAcceleration(match[1], acceleration);
        const endTime = applyAcceleration(match[2], acceleration);
        return `${startTime} --> ${endTime}`;
      }
      return line;
    });
    await fs.writeFile(outputPath, newLines.join('\n'), 'utf8');
  } catch (error) {
    console.error(`Error processing subtitles: ${error.message}`);
  }
}

function calculateWordTimings(subtitles) {
  return subtitles.flatMap(subtitle => {
    const words = subtitle.text.split(' ');
    const totalDuration = subtitle.end - subtitle.start;
    const durationPerLetter = totalDuration / subtitle.text.length; // Including punctuation

    let currentStart = subtitle.start;
    return words.map(word => {
      const wordDuration = word.length * durationPerLetter;
      const wordEnd = currentStart + wordDuration;

      const wordTiming = {
        start: currentStart,
        end: wordEnd,
        word: he.decode(word.replace(',dot,', '.'))
      };

      currentStart = wordEnd;
      return wordTiming;
    });
  });
}

async function readFileAndParse(filePath) {
  try {
    const data = await fs.readFile(filePath, 'utf8');
    return parseSRT(data);
  } catch (error) {
    console.error(`Error reading or parsing file: ${error.message}`);
    return [];
  }
}

function parseSRT(data) {
  const srtRegex = /(\d+)\n(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})\n([\s\S]*?)(?=\n\n|\n*$)/g;
  const result = [];
  let match;

  while ((match = srtRegex.exec(data)) !== null) {
    const formattedText = match[4].replace(/([.,!?])([^\s])/g, '$1 $2').replace(/\n/g, ' ');
    
    result.push({
      start: convertTimeToSeconds(match[2]),
      end: convertTimeToSeconds(match[3]),
      text: formattedText
    });
  }

  return result;
}

function convertTimeToSeconds(time) {
  const [hours, minutes, seconds] = time.split(':');
  const [secs, millis] = seconds.split(',');
  return parseInt(hours, 10) * 3600 + parseInt(minutes, 10) * 60 + parseInt(secs, 10) + parseInt(millis, 10) / 1000;
}

const run = async ({folderPath, acceleration}) => {
  try {
    const inputFilePath = path.join(folderPath, 'subtitles.srt');
    const outputFilePath = path.join(folderPath, 'subtitles_speed.srt');

    await changeSubtitleTiming(inputFilePath, outputFilePath, acceleration);
    console.log('Subtitles timing adjusted.');

    const jsonArray = await readFileAndParse(outputFilePath);
    if (jsonArray.length > 0) {
      const words = calculateWordTimings(jsonArray);
      const outputJsonPath = path.join(folderPath, 'transcription.json');
      await fs.writeFile(outputJsonPath, JSON.stringify(words), 'utf8');
      console.log('Transcription and word timings saved.');
    } else {
      console.log('No subtitles found or failed to parse.');
    }
  } catch (error) {
    console.error(`Error in run process: ${error.message}`);
  }
};

module.exports = {
  run
}