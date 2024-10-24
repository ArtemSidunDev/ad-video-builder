const fs = require('fs')

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

async function main() {
  const dataFolder = './data'
  const count = await getFolderCount(dataFolder)
  console.log(count)
  return count
}

main()