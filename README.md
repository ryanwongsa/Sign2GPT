# Sign2GPT: Leveraging Large Language Models for Gloss-Free Sign Language Translation

Implementation of Sign2GPT. 

The code is currently being updated, the model and loss function has already been uploaded, therefore it is possible to use the model and loss function in any custom training setup in pytorch. If you have any questions, submit an issue in this repo and I will be happy to help.

# TODO

```
[x] Model 
[x] Pseudo-gloss Loss function
[x] Evaluation Metrics
[x] Pseudo-gloss generation script
[x] Dataloader
[ ] Pseudo-gloss pretraining script
[ ] Downstream training script
[ ] Evaluation
```

# CHANGELOGS


[16/01/2025] Added Template training code base

[15/01/2025] Added Dataloader for csldaily

[14/01/2025] Added Pseudo-gloss generation scripts

[04/01/2025] Merged and added the model setup, with evaluation metrics used. 

[03/09/2024] Added decoder with adaptors

[27/08/2024] Added embedding layers

[20/08/2024] Added pretraining head component

[20/08/2024] Added pretraining loss function

[19/07/2024] Added missing transformer encoder modules.

[02/06/2024] Added spatial model architecture setup.

[25/05/2024] Added sign encoder model architecture setup.

[07/04/2024] Created Repo for the paper titled "Sign2GPT: Leveraging Large Language Models for Gloss-Free Sign Language Translation", Work in progress.
