# Contribution Guide

Trace is an actively growing project and under active maintenance and development! We welcome your contributions and involvement. Below are instructions for how to contribute to Trace.

## Quick Bug Fix

If there is a minor, isolated bug that can be directly fixed, the bug fix should be submitted as a pull request and will be merged into the `main` branch. 

## Contributing Feature

We welcome new ideas. 

### Step 1: Feature Spec Doc 
A feature should first be written as a Google Doc (an example is [here](https://docs.google.com/document/d/1FX1ygc8lgFpFn3ni3E2A_DCGtn505PpAM8QaAjEovsA/edit?usp=sharing)).

### Step 2: Create an Issue
An issue should be created, and under the issue, the doc is linked. People should be allowed to comment on the doc.

### Step 3: Implement Feature
Create a separate branch, extending from the `experimental` branch. This branch contains all the new features that have not been merged into the `main` branch yet. 
Make sure your features are implemented, along with `unit tests` or `examples` to show how it's used.

### Step 4: Create a Pull Request
Create a pull request formally to merge into the experiment branch and request a review. This is a lightweight review that only checks for malicious code, or if the feature does not pass its own unit tests.

### Step 5: Merge into Experimental
Once the request is approved, it will be merged into the `experimental` branch.
