# Glossary

This page defines the project terms. Each term has one meaning in all the documents.

| Term | Meaning |
|---|---|
| benchmark | The fixed set of items, the prompt, and the rules that score a model. |
| item | One sentence with one label. |
| source item | One item that has no language. It has a `source_item_id`. |
| language split | The 300 items of one language. |
| label | The correct answer for an item: `yes` or `no`. |
| `yes` | The sentence has land-use, land-cover, or geographic-environment signal. A satellite could see it. |
| `no` | The sentence is only about history, administration, people, or events. |
| positive class | The label `yes`. |
| prompt | The English text that the model receives. The file is `data/prompt.txt`. |
| model | A language model on the roster. |
| roster | The list of models in `domain/roster.py`. |
| generative model | A model that writes text. The benchmark reads a verdict from the text. |
| scoring model | A model that gives a relevance score for each item. It does not write text. |
| run | One model on one language split. |
| run file | The JSON file that holds the data of one run. |
| checkpoint | A run file that is complete. A job skips a run that has a checkpoint. |
| verdict | The `yes` or `no` that the benchmark reads from a generation. |
| generation | The text that a model writes for one item. |
| unparsed | A generation that has no verdict. The benchmark counts it as an error. |
| truncated | A generation that used the complete token budget and did not stop. The benchmark scores it as unparsed. |
| token budget | The maximum number of new tokens in a generation. The published value is 4096. |
| greedy decoding | A decoding mode that always selects the most likely token. The same input gives the same output. |
| `unparsed_rate` | The part of the items that have no verdict. |
| relevance score | The number from a scoring model. A high value means `yes`. |
| threshold sweep | A test of many threshold values on the relevance scores. |
| MCC | Matthews correlation coefficient. |
| runtime | The software that runs a model: Transformers, llama.cpp, or SGLang. |
| SGLang | A runtime that serves a model to many requests. |
| DSpark | A draft model for speculative decoding. |
| speculative decoding | A method in which a draft model proposes tokens and the target model checks them. |
| target model | The model that gives the final output in speculative decoding. |
| draft model | The small model that proposes tokens in speculative decoding. |
| lossless check | A comparison that shows that speculative decoding gives the same output as the target model alone. |
| throughput mode | An SGLang mode that sends many requests at the same time. |
| layer | One of the code groups: `cli`, `application`, `adapters`, `domain`. |
| protocol | A typed interface in the code. `TextGenerator` and `LabelScorer` are protocols. |
| ADR | Architecture decision record. |
| Grid'5000 | The French research testbed that runs the sweep. |
| node | One computer of Grid'5000 that has the GPUs. |
| frontend | The Grid'5000 computer for checkout, submission, and monitoring. |
| shard | One part of the model-language pairs that one node runs. |
| walltime | The maximum time of a job. |
