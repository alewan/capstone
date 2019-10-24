# capstone
Capstone Project for 2019-2020

### envs
Contains the conda environment for the project

### scripts
Contains useful scripts

## Commit Practices

### Merge process
1. ALWAYS REBASE - if unsure how, ask

### Good commit messages
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
