Feature: Command-line benchmark workflows
  Scenario: Run, report and preview a publication without contacting the Hub
    Given a ready CLI harness
    When I run "stub/gold" through lrb
    And I report through lrb
    Then the leaderboard CSV contains the one-row perfect run
    When I preview publication through lrb
    Then the preview lists the run and makes no Hub call

  Scenario: Resume run-all without regenerating an existing result
    Given a ready CLI harness
    And a completed result for "stub/first"
    When I resume run-all for "stub/first" and "stub/second"
    Then only "stub/second" is generated

  Scenario: Report names a corrupted stored result
    Given a ready CLI harness
    And a truncated result file
    When I report through lrb
    Then reporting fails and names the truncated file

  Scenario: Report verifies a lossless speculative run
    Given a ready CLI harness
    And identical baseline and speculative results
    When I report through lrb
    Then the report states that the speculative run is lossless

  Scenario Outline: Report refuses mixed benchmark settings
    Given a ready CLI harness
    And stored runs with a mismatched "<setting>" digest
    When I report through lrb
    Then reporting refuses the mixed settings

    Examples:
      | setting   |
      | benchmark |
      | prompt    |
