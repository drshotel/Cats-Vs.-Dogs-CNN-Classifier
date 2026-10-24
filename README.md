# Cats vs. Dogs Image Classification with a CNN (PyTorch)

A convolutional neural network built from scratch in PyTorch to classify images as cats or dogs. Unlike my previous CIFAR-10 project, this one was written entirely by hand — no AI-generated code — with AI used only to explain unfamiliar concepts (e.g. `ImageFolder`, GPU device placement) and help debug errors, never to write the model or training code itself.

## Self-Imposed Rules

Before starting this project, I set the following constraints for myself, since my previous project relied on an AI-scaffolded architecture:

- Usage of AI in order to learn and understand foreign concepts is allowed.
- Using AI in order to write code for me is not allowed.
- Avoid copy and pasting code.
- Only type the code if I actually understand what it's doing.

The goal was to prove I could build a CNN myself, not just run and tune one someone else wrote.

## Final Result

| Set | Accuracy |
|---|---|
| Train | 90.83% |
| Validation | 88.92% |
| Test | 88.92% |

Train, validation, and test accuracy all closely agree, which confirms the final model generalizes well rather than being overfit to the training data or to tuning decisions made during development.

## Experiment Log

Each change was tested one at a time, starting from a freshly initialized model, so results could be attributed to a single cause.

### 1. Baseline (2 conv layers)
- 5 epochs → 80.18% validation accuracy
- 15 epochs → ~81% validation accuracy
- Confirmed more epochs alone weren't improving results much — the model had plateaued. Loss curve was noticeably unstable (bouncing rather than smoothly decreasing).

### 2. + Batch Normalization
- 15 epochs → 80.56% validation accuracy — no meaningful improvement
- Loss curve was *more* volatile with batch norm than without it
- **Conclusion:** dropped from the final model. Didn't help on this architecture/dataset combination.

### 3. + Extra Conv Layer (3 conv layers total)
- Required re-calculating the flatten size for the fully connected layer after the architecture change — found the correct value (4608) by running a dummy input through just the convolutional layers and printing the output shape, rather than computing it by hand.
- 15 epochs → 82.5–82.8% validation accuracy — a real improvement over baseline
- Loss curve showed a clean, steady decrease through roughly epoch 8, then more volatility afterward.

### 4. Epoch Count Check
- Compared results at ~8 epochs vs. 15 epochs — nearly identical validation accuracy, confirming the model's performance plateaus early and later epochs weren't adding much.

### 5. Lower Learning Rate (0.0001 instead of 0.001)
- 15 epochs → 84.30% validation accuracy — best result up to this point
- Checked training accuracy for the first time and found **97.30% train vs. 84.30% validation** — a ~13-point gap indicating significant overfitting.

### 6. Data Augmentation — First Attempt (bug)
- Result was invalid: training was accidentally resumed on an already-trained model instead of starting from fresh random weights, producing artificially low loss values that didn't reflect what augmentation was actually doing.
- **Lesson:** always reinitialize the model when testing a change to the data pipeline, so results are comparable to a clean baseline.

### 7. Data Augmentation — Second Attempt (bug)
- Train 98.41%, validation 83.66% — the overfitting gap didn't improve
- Root cause: the same `transform` (including random rotation and horizontal flip) was being applied to the validation/test set as well as the training set, corrupting the evaluation.
- **Lesson:** augmentation should only ever be applied to training data. Validation and test data should remain clean and consistent for an honest accuracy reading.

### 8. Data Augmentation — Fixed
- Used a separate `train_transform` (resize, random rotation, random horizontal flip, normalization) for training data, and a clean `test_transform` (resize, normalization only) for validation and test data.
- 15 epochs → **train 90.50%, validation 87.98%** — the overfitting gap closed from ~13 points down to ~2.5 points, and validation accuracy improved to the best result so far.

### 9. + Dropout (p=0.3, applied after flattening, before the final linear layer)
- 15 epochs → **train 90.83%, validation 88.92%, test 88.92%** — final model
- Small additional improvement with the overfitting gap staying tight (~2 points).

## Key Takeaways

- **Batch normalization isn't universally helpful** — it didn't improve this architecture and even increased training instability.
- **Checking train accuracy alongside validation accuracy is essential** — without it, the ~13-point overfitting gap after learning rate tuning would have gone unnoticed.
- **Data augmentation must be applied correctly** — accidentally augmenting the validation/test set, or continuing training on an already-overfit model, both produced misleading results that looked promising at first glance.
- **Systematic, one-change-at-a-time testing** made it possible to actually attribute each accuracy change to its cause, rather than guessing.

## Tech Stack

- PyTorch (model architecture, training loop, autograd)
- torchvision (`ImageFolder`, transforms)
- Google Colab (GPU training)

## Debugging Along the Way

Writing this from scratch surfaced several real bugs that a pre-built architecture wouldn't have exposed:

- **DataLoader setup with `ImageFolder`** — hadn't used `ImageFolder` before. Used AI to understand how it expects folder structure and how to wire it into a DataLoader correctly, then implemented and adapted it myself for this dataset.
- **Shape mismatch after adding a layer** — after adding a third convolutional layer, the model threw a channel-mismatch error. Diagnosed it by tracing the error message back to a forward-pass line that was mistakenly reusing `self.conv2` twice instead of calling `self.conv3`.
- **Device mismatch during shape probing** — got a `RuntimeError` from mixing a GPU-resident model with a CPU-resident dummy tensor while probing the flatten size. Fixed by moving the dummy tensor to the same device as the model.
- **Stale model reused across experiments** — an early data augmentation test produced suspiciously low loss values. Traced this to training being resumed on a model that had already memorized the training set, rather than starting from a freshly initialized one. Fixed by explicitly recreating the model and optimizer before each new experiment.
- **Shared transform across train and test sets** — a second augmentation attempt still showed no improvement in the overfitting gap. Found that the same `transform` object (including random rotation and flip) was being applied to validation/test data as well as training data, corrupting the evaluation. Fixed by splitting into separate `train_transform` and `test_transform` pipelines.

## Infrastructure Notes

- Loading images directly from Google Drive during training was a major bottleneck (~30 seconds per batch). Copying the dataset to Colab's local storage before training reduced this to ~2 seconds per batch.
- Training initially ran on CPU despite GPU runtime being enabled, because the model and data tensors were never explicitly moved to the GPU device. Adding `.to(device)` calls reduced epoch time from ~55–60 minutes to ~3 minutes.

## What I'd Try Next

- Model capacity adjustments (more/fewer filters per layer)
- Additional augmentation types (color jitter, random crop)
- A deeper architecture or transfer learning from a pretrained model, to compare against this from-scratch approach
