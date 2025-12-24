## Power package Installation
Dedicated package for AI component. This must be docker composed in the future.

There are two ways to use this package in the host server:
- Serverless and API calls which runs inference fully remotely -> Use `reliable-requirements.txt` to pip install dependencies.
- Local inference in any way (Not tested) (e.g. Whisper model with transformer pipeline but other parts are remote calls) -> Use `cuda-requirements.txt` to pip install. (Apple Silicons and TPUs are not supported at this time) If you wish for a CPU only local inference, you should install required library manually. (e.g. torch, transformers)

### Environment Variables
For Langchain calls, include `LANGCHAIN_BASE_URL`.

### Serverless
Serverless platform is powered by Modal (https://modal.com/docs).<br>
For the first time configuring, run following:
```
$ modal setup
```
The main script to run on remote is `power/inference/serverless.py`<br>
To deploy any changes, run following:
```
$ modal deploy backend/power/inference/serverless.py
```

### Power Package Front-end
Front-end in this package is powered by React with Vite and just for development purposes.<br>
Make sure you have `npm >= 10.0` then run:
```
$ cd backend/power/webui
$ npm install
```

This front-end and Django back-end is proxied via Caddy. You can get binary here and place it under the `webui` folder.
(https://github.com/caddyserver/caddy/releases/tag/v2.10.2)<br>
Then run following to start front-end:
```
$ cd backend/power/webui
$ ./caddy run
```
```
$ cd backend/power/webui
$ npm run dev
```
To access, follow this link: http://localhost:8795

### Back-end
Back-end is managed by Django and there is no specific configuration must be done for this. Follow the README there.
