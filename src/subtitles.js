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

function changeSubtitleTiming(filePath, outputPath, acceleration) {
  const fileContent = fs.readFileSync(filePath, 'utf8');
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
  fs.writeFileSync(outputPath, newLines.join('\n'), 'utf8');
}

function calculateWordTimings(subtitles) {
  const wordTimings = [];
  subtitles.forEach(subtitle => {
    const words = subtitle.text.split(' ');
    const totalDuration = subtitle.end - subtitle.start;
    const durationPerLetter = totalDuration / subtitle.text.length;

    let currentStart = subtitle.start;

    words.forEach(word => {
      const wordDuration = word.length * durationPerLetter;
      const wordEnd = currentStart + wordDuration;

      wordTimings.push({
        start: currentStart,
        end: wordEnd,
        word: word
      });

      currentStart = wordEnd;
    });
  });

  return wordTimings;
}

function readFileAndParse(filePath, folderPath) {
    const data = fs.readFileSync(filePath)

    const jsonArray = parseSRT(data);

    return jsonArray;
}

function parseSRT(data) {
  const srtRegex = /(\d+)\n(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})\n([\s\S]*?)(?=\n\n|\n*$)/g;
  const result = [];
  let match;

  while ((match = srtRegex.exec(data)) !== null) {
    result.push({
      start: convertTimeToSeconds(match[2]),
      end: convertTimeToSeconds(match[3]),
      text: match[4].replace(/\n/g, ' ')
    });
  }

  return result;
}

function convertTimeToSeconds(time) {
  const [hours, minutes, seconds] = time.split(':');
  const [secs, millis] = seconds.split(',');
  return parseInt(hours, 10) * 3600 + parseInt(minutes, 10) * 60 + parseInt(secs, 10) + parseInt(millis, 10) / 1000;
}

const run = async (data) => {
  const { 
    folderPath, 
    acceleration 
  } = data;

  await changeSubtitleTiming(`${folderPath}/subtitles.srt`, `${folderPath}/subtitles_speed.srt`, acceleration)
  const inputFilePath = `${folderPath}/subtitles_speed.srt`;

  const jsonArray = readFileAndParse(inputFilePath, folderPath);
  
  const words = calculateWordTimings(jsonArray);
  
  fs.writeFileSync(`${folderPath}/transcription.json`, JSON.stringify(words), { encoding: 'utf8' });
};

module.exports = {
  run
}