- refactor:
  - [ ] write preprocessors for classification and localization
  - [ ] fix classification dataset bug
  - [ ] rewrite examples
  - [ ] create masks for `tier3` data
  - [ ] add prediction to validator
  - [ ] create multiple model validator
  - [ ] check why localization model is used separately in classification task  
  - [ ] give weights to cce loss to give layers 2 and 3 more attention

- Test:
  - [ ] test localization models
  - [ ] test classification models
  - [ ] on test and holdout set
  - [ ] predict and calculate accuracy on `test` and `holdout` sets
  
- Improvements On Original Model
  - [ ] train and test localization models with post-disaster images too
  - [ ] experiment with `EfficientNet B0` and more 

- *MAML*:
  - [ ] read about MAML
  - [ ] read *MAML* codes
  - [ ] create tasks from dataset
  - [ ] create meta-batch from a task
  - [ ] overall workflow of maml training
  - [ ] implement *MAML* algorithm

- *Replite*:
  - [ ] read about difference of *MAML* and *Replite*
  - [ ] implement *Reptile* algorithm