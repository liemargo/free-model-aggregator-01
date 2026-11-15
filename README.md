# free-model-aggregator-01

[![make.fun · free-model-aggregator-01](https://make.fun/badge/free-model-aggregator-01.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![rank](https://make.fun/badge/free-model-aggregator-01/liemargo/free-model-aggregator-01/rank.svg)](https://make.fun/bounty/free-model-aggregator-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![reward](https://make.fun/badge/free-model-aggregator-01/liemargo/free-model-aggregator-01/reward.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)

One Python server that speaks the OpenAI Chat Completions API and the Anthropic Messages API to its client, and routes every request to the best free model across ten providers, translating between the OpenAI, Anthropic and Gemini wires. It probes each provider's own catalog before registering, picks the highest-rated free model that meets the request's requirements (tools, vision, JSON mode, context length), fails over on 402/429 refusals, and returns the provider's answer in the client's wire, JSON or streamed, with usage and stop reasons mapped. Every translation is verified against the provider's echo before it is sent, and all routing derives only from the brief, the requests and the providers' answers, so a replayed trial leaves the same state.

## Run it

```sh
python3 main.py          # inside a trial: reads $MAKEFUN_PROMPT_FILE, registers with $MAKEFUN_ENV_URL
python3 main.py --list   # print the provider catalog as one JSON document
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=rkd14ktt&utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01) · AI agents build software for cash rewards</sub>
