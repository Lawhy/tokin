<h1 align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/header-dark.svg">
    <img src="assets/header-light.svg" alt="tokin. Make any agent harness token-native." width="480">
  </picture>
</h1>

<p align="center">
  <a href="https://github.com/open-rollout/tokin/actions/workflows/ci.yml"><img src="https://github.com/open-rollout/tokin/actions/workflows/ci.yml/badge.svg?branch=main" alt="CI"></a>
  <a href="https://pypi.org/project/tokin/"><img src="https://img.shields.io/pypi/v/tokin" alt="PyPI"></a>
  <a href="LICENSE"><img src="https://img.shields.io/github/license/open-rollout/tokin" alt="License"></a>
</p>

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/gateway-dark.svg">
    <img src="assets/gateway-light.svg" alt="tokin as a gateway. Harnesses, any harness unchanged, each call their own session URL with chat completions in text. Inside tokin each turn is attached to its place in its session's tree; rendered, keeping the ids the model sampled for rollouts, rollout and s, where re-encoding the text would split it as roll and outs; generated on the inference engine from input tokens to output tokens; and parsed back into an assistant message, {&quot;role&quot;: &quot;assistant&quot;, …}, with reasoning and tool calls for its harness. The turn's ids grow the session's tree at that place, each leaf a rollout with token_ids, loss_mask and logprobs, which a trainer collects to update the weights the engine serves." width="100%">
  </picture>
</p>

**tokin** is a gateway between an agent harness and an inference engine. The harness
keeps speaking text over a standard chat-completions API; the engine only ever receives
token ids. In between, tokin keeps each sample's session as a tree of turns, renders
every prompt by appending to the ids the model already sampled instead of re-encoding
text, and parses each generation back into an assistant message with its reasoning and
tool calls. Each leaf of the tree is a rollout a trainer can use as is.
