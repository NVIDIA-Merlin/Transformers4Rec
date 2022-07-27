# Contributing to Transformers4Rec

If you are interested in contributing to Transformers4Rec your contributions will fall
into three categories:
1. You want to report a bug, feature request, or documentation issue
    - File an [issue](https://github.com/NVIDIA-Merlin/Transformers4Rec/issues/new/choose)
    describing what you encountered or what you want to see changed.
    - The NVIDIA-Merlin team will evaluate the issues and triage them, scheduling
    them for a release. If you believe the issue needs priority attention
    comment on the issue to notify the team.
2. You want to propose a new Feature and implement it
    - Post about your intended feature, and we shall discuss the design and
    implementation.
    - Once we agree that the plan looks good, go ahead and implement it, using
    the [code contributions](#code-contributions) guide below.
3. You want to implement a feature or bug-fix for an outstanding issue
    - Follow the [code contributions](#code-contributions) guide below.
    - If you need more context on a particular issue, please ask and we shall
    provide.

## Code contributions

### Your first issue

1. Read the project's [README.md](https://github.com/NVIDIA-Merlin/Transformers4Rec/blob/main/README.md)
    to learn how to setup the development environment
2. Find an issue to work on. The best way is to look for the [good first issue](https://github.com/NVIDIA-Merlin/Transformers4Rec/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22)
    or [help wanted](https://github.com/NVIDIA-Merlin/Transformers4Rec/issues?q=is%3Aissue+is%3Aopen+label%3A%22help+wanted%22) labels
3. Comment on the issue saying you are going to work on it
4. Code! Make sure to update unit tests!
5. When done, [create your pull request](https://github.com/NVIDIA-Merlin/Transformers4Rec/compare)
6. Verify that CI passes all [status checks](https://help.github.com/articles/about-status-checks/). Fix if needed
7. Wait for other developers to review your code and update code as needed
8. Once reviewed and approved, a developer will merge your pull request

Remember, if you are unsure about anything, don't hesitate to comment on issues
and ask for clarifications!

### Seasoned developers

Once you have gotten your feet wet and are more comfortable with the code, you
can look at the prioritized issues of our next release in our [project boards](https://github.com/NVIDIA-Merlin/Transformers4Rec/projects).

> **Pro Tip:** Always look at the release board with the highest number for
issues to work on. This is where Transformers4Rec developers also focus their efforts.

Look at the unassigned issues, and find an issue you are comfortable with
contributing to. Start with _Step 3_ from above, commenting on the issue to let
others know you are working on it. If you have any questions related to the
implementation of the issue, ask them in the issue instead of the PR.

## Label your PRs

This repository uses the release-drafter action to draft and create our change log.

Please add one of the following labels to your PR to specify the type of contribution
and help categorize the PR in our change log:

- `breaking` -- The PR creates a breaking change to the API.
- `bug` -- The PR fixes a problem with the code.
- `feature` or `enhancement` -- The PR introduces a backward-compatible feature.
- `documentation` or `examples` -- The PR is an addition or update to documentation.
- `build`, `dependencies`, `chore`, or `ci` -- The PR is related to maintaining the
  repository or the project.

By default, an unlabeled PR is listed at the top of the change log and is not
grouped under a heading like *Features* that groups similar PRs.
Labeling the PRs so we can categorize them is preferred.

If, for some reason, you do not believe your PR should be included in the change
log, you can add the `skip-changelog` label.
This label excludes the PR from the change log.

For more information, see `.github/release-drafter.yml` in the repository
or go to <https://github.com/release-drafter/release-drafter>.

## Attribution

Portions adopted from https://github.com/pytorch/pytorch/blob/master/CONTRIBUTING.md
