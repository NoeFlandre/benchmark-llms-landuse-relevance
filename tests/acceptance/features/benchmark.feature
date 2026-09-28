Feature: Benchmarking small LLMs on land-use relevance
  As a researcher comparing small language models,
  I want each model scored on the active multilingual labelled sentences with the same prompt,
  so that the leaderboard reflects the models and not the harness.

  Background:
    Given the project benchmark of labelled sentences
    And the project prompt template

  Scenario: A model that follows the output contract perfectly
    Given a model that answers every sentence with its gold label
    When I benchmark that model
    Then the run scores an accuracy of 1.00
    And no generation is left unparsed
    And the stored result covers every sentence in the benchmark

  Scenario: A model that always answers yes
    Given a model that always answers "yes"
    When I benchmark that model
    Then the run recalls every relevant sentence
    And the run misses no relevant sentence

  Scenario: A model that ignores the output contract
    Given a model that always answers "I cannot decide"
    When I benchmark that model
    Then every generation is left unparsed
    And the run scores an accuracy of 0.00
    And the raw generations are kept in the stored result

  Scenario: A model cut off before it reaches a verdict
    Given a model whose answer is cut off by the token budget
    When I benchmark that model
    Then every generation is left unparsed
    And the stored result marks every generation as truncated
    And the run scores an accuracy of 0.00

  Scenario: Comparing models on a leaderboard
    Given a model that answers every sentence with its gold label
    And a second model that always answers "no"
    When I benchmark both models
    Then the leaderboard ranks the accurate model first
    And the leaderboard has one row per model

  Scenario: The run records exactly which inputs produced it
    Given a model that answers every sentence with its gold label
    When I benchmark that model
    Then the stored result pins the benchmark and prompt digests
    And the stored result names the model revision that was used

  Scenario: The active multilingual inventory is complete and aligned
    Given the active multilingual benchmark inventory
    Then 85 languages are available with 300 rows each
    And every language shares the same source identity sequence

  Scenario: Model-language checkpoints aggregate without collisions
    Given a gold model run for each of two languages
    When I read the active results
    Then each language has its own checkpoint
    And the model aggregate covers both languages

  Scenario: Archived results are outside the active contract
    Given an active result directory with an archived invalid file
    When I read the active archive directory
    Then the archived file is ignored

  Scenario: Deterministic shards cover the multilingual sweep exactly once
    Given a multilingual roster and four shard slots
    Then the shard union is complete and disjoint
