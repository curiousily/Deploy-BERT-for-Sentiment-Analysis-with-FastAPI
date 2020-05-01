# Deploy BERT for Sentiment Analsysi with FastAPI

Deploy a pre-trained BERT model for Sentiment Analysis as a REST API using FastAPI

## Installation

Clone this repo:

```sh
git clone git@github.com:curiousily/Deploy-BERT-for-Sentiment-Analysis-with-FastAPI.git
cd Deploy-BERT-for-Sentiment-Analysis-with-FastAPI
```

Install the dependencies:

```sh
pipenv install --dev
```

Download the pre-trained model:

```sh
bin/download_model
```

## Test the setup

Start the HTTP server:

```sh
bin/start_server
```

Send a test request:

```sh
bin/test_request
```

## License

MIT