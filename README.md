This repository (cleaned) removes the large raw dataset files.

Large files removed:
- amazoncat13k/AmazonCat-13K.raw/trn.json.gz
- amazoncat13k/AmazonCat-13K.raw/tst.json.gz

To host the dataset, upload those two files to S3, Google Drive, or GitHub Releases and update the download script.

Download instructions (example using curl):

# replace URL with the file URL
curl -L -o trn.json.gz "<TRN_URL>"
curl -L -o tst.json.gz "<TST_URL>"

When the files are placed beside this repo, the code can reference them at `amazoncat13k/AmazonCat-13K.raw/`.

