<!--
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
-->

# Contributing to KernelBench Verified
We want to make contributing to this project as easy and transparent as
possible.

## Our Development Process
KernelBench Verified is developed jointly by Meta and Stanford University, and its source code is synced out to this public GitHub repository. Because of this, our primary source of truth is our internal version control system, and the GitHub repository acts as a mirror.

To accommodate this setup, our pull request (PR) workflow is slightly different than a standard GitHub-only project. When you submit a PR on GitHub, it will go through the following process:

1. **Review:** A project maintainer will review your code on GitHub and provide feedback.
2. **Import:** Once your PR is approved, a maintainer will import your changes into Meta's internal systems. 
3. **Internal Testing:** We will run your changes against our internal test suites and continuous integration (CI) pipelines. 
4. **Merge & Sync:** Upon passing all internal checks, your code will be merged internally. Shortly after, an automated sync process will push the merged commit back to the public GitHub repository. 

**Note:** This automated sync process will close your original GitHub PR automatically (rather than showing as "Merged" via the GitHub UI), but the commit history will still properly attribute the changes to you. Because of this internal testing phase, it may take a little extra time for approved PRs to show up in the `main` branch. We appreciate your patience!

## Pull Requests
We actively welcome your pull requests.

1. Fork the repo and create your branch from `main`.
2. If you've added code that should be tested, add tests.
3. If you've changed APIs, update the documentation.
4. Ensure the test suite passes.
5. Make sure your code lints.
6. If you haven't already, complete the Contributor License Agreement ("CLA").

## Contributor License Agreement ("CLA")
In order to accept your pull request, we need you to submit a CLA. You only need
to do this once to work on any of Meta's open source projects.

Complete your CLA here: <https://code.facebook.com/cla>

## Issues
We use GitHub issues to track public bugs. Please ensure your description is
clear and has sufficient instructions to be able to reproduce the issue.

Meta has a [bounty program](https://bugbounty.meta.com/) for the safe
disclosure of security bugs. In those cases, please go through the process
outlined on that page and do not file a public issue.

## Coding Style  
* 2 spaces for indentation rather than tabs
* 80 character line length
* ...

## License
By contributing to KernelBench Verified, you agree that your contributions will be licensed
under the LICENSE file in the root directory of this source tree.
