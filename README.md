# capstone
2019-2020 Capstone Project

## Directory Structure

### envs
Contains the conda environment for the project

### scripts
Contains useful scripts

### aliases.sh
Contains useful aliases for common commands


## Issue Practices
### Creating Issues
1. Decide what type of issue it is and tag it: story, task, etc.
    1. Stories typically follow the INVEST pattern (Independent, Verifiable, Estimable, Small, and Testable) and describe a new feature being added for a particular purpose
        1. All stories need an objective and acceptance critera (AC). Objective is the goal of the story. Acceptance criteria is what is required before a task can be considered completed.
    1. Tasks are the smallest units of work
    1. Bugs are issues with the existing codebase that require fixing
1. Add it to the appropriate Sprint/Milestone
1. Assign it to the appropriate person
1. Add it to the project so that it is visible on the board and can be tracked.

## Commit Practices

### Merge Process
1. ALWAYS REBASE - if unsure how, ask (Note: This is also the only option enabled in GitHub for PRs right now)

### Good Commit Messages
1. 3-4 character tag before commit title (either from existing list of tags, or be created)
    1. Can have multiple tags with space between (PREP NN)
1. Descriptive title of what commit does
1. Commit message with major updates & why they were made
1. Tag the related issue in the commit message</br>
`Issue-Id: #xxx`
1. Update the environment.yaml if you add new dependencies</br>
`conda env export > <path_to_envs>/environment.yaml`
    1. If the environent is updated, add ENV-UP tag to end of commit
1. Limit characters per line to 70 for readability


#### Example Commit
```
PREP Environment Preparation

- Add a conda environment config in an environments directory
for consistency amongst the team
- Create a directory for scripts

Issue-Id: #69
ENV-UP
```

### List of current tags
- PREP - Initial preparation work for the project
- NN - neural net
- PROC - processing
- FE - frontend
- BE - backend
- IMG - images/video 
- AUD - audio
